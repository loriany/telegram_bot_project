import logging
import os
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

import torch
from telegram import Update
from telegram.constants import ChatAction
from telegram.ext import (
    Application,
    CommandHandler,
    ContextTypes,
    MessageHandler,
    filters,
)
from torch import nn
from transformers import AutoConfig, AutoModel, AutoModelForTokenClassification, AutoTokenizer

import gc

os.environ["TOKENIZERS_PARALLELISM"] = "false"
torch.set_num_threads(1)

# =========================
# Configuration
# =========================

from dotenv import load_dotenv
load_dotenv()

BASE_DIR = Path(__file__).resolve().parent
MODELS_DIR = BASE_DIR / "models"
SPAN_MODEL_DIR = Path(os.getenv("SPAN_MODEL_DIR", str(MODELS_DIR / "spans")))
TECH_CHECKPOINT_PATH = Path(
    os.getenv("TECH_CHECKPOINT_PATH", str(MODELS_DIR / "tech" / "best_model.pt"))
)
TELEGRAM_BOT_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN", "")
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

SPAN_MAX_LEN = 512
SPAN_MERGE_DISTANCE = 0
TECH_MAX_LEN = 512
TECH_MAX_LABELS = 3

TECHNIQUES = [
    "straw_man",
    "appeal_to_fear",
    "fud",
    "bandwagon",
    "whataboutism",
    "loaded_language",
    "glittering_generalities",
    "euphoria",
    "cherry_picking",
    "cliche",
]

TECHNIQUE_MAP_UK = {
    "loaded_language": "Навантажена мова",
    "glittering_generalities": "Блискучі узагальнення",
    "euphoria": "Ейфорія",
    "appeal_to_fear": "Апеляція до страху",
    "fud": "Страх, непевність та сумніви",
    "bandwagon": "Апеляція до народу",
    "cliche": "Кліше без суті",
    "whataboutism": "Сам дурень!",
    "cherry_picking": "Вибіркова правда",
    "straw_man": "Опудало",
}

WORD_CHARS_RE = re.compile(r"[A-Za-zА-Яа-яІіЇїЄєҐґЁё0-9]")
URL_RE = re.compile(r"(https?://|www\.|youtu\.be|t\.me|telegram\.me)", re.IGNORECASE)
STOP_WORDS_SHORT = {
    "і", "й", "та", "а", "у", "в", "на", "по", "не", "но", "или", "и",
    "the", "of", "to", "in", "on", "at", "for",
}

logging.basicConfig(
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    level=logging.INFO,
)
logger = logging.getLogger(__name__)


# =========================
# Technique model
# =========================
class ManipulationClassifier(nn.Module):
    def __init__(self, model_name: str, num_labels: int, dropout_rate: float = 0.3):
        super().__init__()
        self.model = AutoModel.from_pretrained(model_name)
        self.dropouts = nn.ModuleList([nn.Dropout(dropout_rate) for _ in range(5)])
        hidden_size = self.model.config.hidden_size
        self.pre_classifier = nn.Linear(hidden_size, hidden_size)
        self.activation = nn.GELU()
        self.classifier = nn.Linear(hidden_size, num_labels)
        nn.init.xavier_normal_(self.pre_classifier.weight)
        nn.init.xavier_normal_(self.classifier.weight)

    def forward(self, input_ids: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        outputs = self.model(input_ids=input_ids, attention_mask=attention_mask)
        sequence_output = outputs[0]
        cls_output = sequence_output[:, 0, :]
        cls_output = self.pre_classifier(cls_output)
        cls_output = self.activation(cls_output)

        logits = torch.zeros(cls_output.size(0), len(TECHNIQUES), device=cls_output.device)
        for dropout in self.dropouts:
            logits += self.classifier(dropout(cls_output))
        logits = logits / len(self.dropouts)
        return logits


@dataclass
class TechniquePrediction:
    labels_en: List[str]
    labels_uk: List[str]
    scores: Dict[str, float]


class TechniqueService:
    def __init__(self, checkpoint_path: Path, device: torch.device):
        self.checkpoint_path = checkpoint_path
        self.device = device
        self.checkpoint = torch.load(str(checkpoint_path), map_location=device, weights_only=False)
        self.config = self.checkpoint.get("config", {})
        self.model_name = self.config.get("model_name", "xlm-roberta-large")
        self.max_length = int(self.config.get("max_length", TECH_MAX_LEN))
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        self.model = ManipulationClassifier(
        model_name=self.model_name,
        num_labels=len(TECHNIQUES),
        dropout_rate=float(self.config.get("dropout_rate", 0.3)),
        )
        self.model.eval()
        self.model.load_state_dict(self.checkpoint["model_state_dict"])
        for param in self.model.parameters():
            param.requires_grad = False
        self.model.to(device)
        self.model.eval()
        raw_thresholds = self.checkpoint.get("thresholds") or {}
        self.thresholds = {k: float(v) for k, v in raw_thresholds.items()}
        logger.info("Technique model loaded from %s", checkpoint_path)

    @staticmethod
    def clean_text(text: str) -> str:
        if not isinstance(text, str):
            return ""
        text = re.sub(r"https?://\S+|www\.\S+", "[URL]", text)
        text = re.sub(r"\s+", " ", text).strip()
        return text

    def predict_one(self, text: str, max_labels: int = TECH_MAX_LABELS) -> TechniquePrediction:
        clean = self.clean_text(text)
        encoding = self.tokenizer(
            clean,
            max_length=self.max_length,
            padding="max_length",
            truncation=True,
            return_tensors="pt",
        )
        input_ids = encoding["input_ids"].to(self.device)
        attention_mask = encoding["attention_mask"].to(self.device)

        with torch.no_grad():
            logits = self.model(input_ids=input_ids, attention_mask=attention_mask)
            probs = torch.sigmoid(logits).squeeze(0).cpu().tolist()

        scores = {tech: float(prob) for tech, prob in zip(TECHNIQUES, probs)}
        candidates = []
        for tech in TECHNIQUES:
            thr = float(self.thresholds.get(tech, 0.5))
            if scores[tech] >= thr:
                candidates.append((tech, scores[tech]))

        # Найсильніші зверху
        candidates = sorted(candidates, key=lambda x: x[1], reverse=True)

        # Залишаємо тільки top-k
        candidates = candidates[:max_labels]

        # Додатковий фільтр: не брати техніки, які занадто слабкі відносно найкращої
        if candidates:
            best_score = candidates[0][1]
            candidates = [item for item in candidates if item[1] >= best_score - 0.10]

        labels_en = [tech for tech, _ in candidates]
        labels_uk = [TECHNIQUE_MAP_UK[label] for label in labels_en]

        return TechniquePrediction(labels_en=labels_en, labels_uk=labels_uk, scores=scores)


# =========================
# Span model
# =========================
@dataclass
class SpanPrediction:
    spans: List[Tuple[int, int]]
    fragments: List[str]
    highlighted_text: str


class SpanService:
    def __init__(self, model_dir: Path, device: torch.device):
        self.model_dir = model_dir
        self.device = device
        self.tokenizer = AutoTokenizer.from_pretrained(str(model_dir))
        self.config = AutoConfig.from_pretrained(str(model_dir))
        self.model = AutoModelForTokenClassification.from_pretrained(str(model_dir), config=self.config)
        self.model.to(device)
        self.model.eval()
        logger.info("Span model loaded from %s", model_dir)

    @staticmethod
    def expand_to_word(text: str, start: int, end: int) -> Tuple[int, int]:
        while start > 0 and WORD_CHARS_RE.match(text[start - 1]):
            start -= 1
        while end < len(text) and WORD_CHARS_RE.match(text[end]):
            end += 1
        return start, end

    @staticmethod
    def trim_span_edges(text: str, start: int, end: int) -> Tuple[int, int]:
        while start < end and text[start].isspace():
            start += 1
        while end > start and text[end - 1].isspace():
            end -= 1

        bad_left = ".,:;!?)]}\"'»…\n\t#*-–—"
        bad_right = "([{\"'«\n\t#*-–—"

        while start < end and text[start] in bad_right:
            start += 1
        while end > start and text[end - 1] in bad_left:
            end -= 1
        return start, end

    @staticmethod
    def is_garbage_span(span_text: str) -> bool:
        span_text = span_text.strip()
        if not span_text:
            return True
        if URL_RE.search(span_text):
            return True
        if not any(ch.isalnum() for ch in span_text):
            return True
        if len(span_text) <= 2:
            return True
        if span_text.lower() in STOP_WORDS_SHORT:
            return True
        if span_text.lower() in {"youtu", "youtu.be", "http", "https", "www", "t.me"}:
            return True
        return False

    @staticmethod
    def merge_processed_spans(
        spans: Sequence[Tuple[int, int]],
        merge_distance: int = 0,
        max_merged_len: int = 180,
    ) -> List[Tuple[int, int]]:
        if not spans:
            return []
        spans = sorted(spans, key=lambda x: x[0])
        merged = [spans[0]]
        for cur_start, cur_end in spans[1:]:
            prev_start, prev_end = merged[-1]
            if cur_start - prev_end <= merge_distance:
                candidate = (prev_start, max(prev_end, cur_end))
                if candidate[1] - candidate[0] <= max_merged_len:
                    merged[-1] = candidate
                else:
                    merged.append((cur_start, cur_end))
            else:
                merged.append((cur_start, cur_end))
        return merged

    def postprocess_spans(
        self,
        text: str,
        spans: Sequence[Tuple[int, int]],
        min_len: int = 3,
        merge_distance: int = 0,
    ) -> List[Tuple[int, int]]:
        processed: List[Tuple[int, int]] = []
        for start, end in spans:
            if not (0 <= start < end <= len(text)):
                continue
            start, end = self.expand_to_word(text, start, end)
            start, end = self.trim_span_edges(text, start, end)
            if not (0 <= start < end <= len(text)):
                continue
            span_text = text[start:end].strip()
            if len(span_text) < min_len:
                continue
            if self.is_garbage_span(span_text):
                continue
            processed.append((start, end))
        processed = sorted(set(processed), key=lambda x: x[0])
        return self.merge_processed_spans(processed, merge_distance=merge_distance)

    @staticmethod
    def tokens_to_char_spans(
        token_preds: Sequence[int],
        offset_mapping: Sequence[Tuple[int, int]],
        merge_distance: int = 0,
    ) -> List[Tuple[int, int]]:
        spans: List[Tuple[int, int]] = []
        current_span: List[int] | None = None
        for pred, (start, end) in zip(token_preds, offset_mapping):
            if start == end == 0:
                continue
            if pred == 1:
                if current_span is not None:
                    spans.append((current_span[0], current_span[1]))
                current_span = [start, end]
            elif pred == 2:
                if current_span is None:
                    current_span = [start, end]
                else:
                    current_span[1] = max(current_span[1], end)
            else:
                if current_span is not None:
                    spans.append((current_span[0], current_span[1]))
                    current_span = None
        if current_span is not None:
            spans.append((current_span[0], current_span[1]))
        spans = [span for span in spans if span[0] < span[1]]
        return SpanService.merge_processed_spans(spans, merge_distance=merge_distance)

    @staticmethod
    def highlight_text(text: str, spans: Sequence[Tuple[int, int]]) -> str:
        highlighted = text
        for start, end in sorted(spans, key=lambda x: x[0], reverse=True):
            highlighted = highlighted[:start] + "[[[" + highlighted[start:end] + "]]]" + highlighted[end:]
        return highlighted

    def predict_one(self, text: str) -> SpanPrediction:
        encoding = self.tokenizer(
            text,
            return_offsets_mapping=True,
            add_special_tokens=True,
            truncation=True,
            max_length=SPAN_MAX_LEN,
            padding=False,
            return_tensors="pt",
        )
        input_ids = encoding["input_ids"].to(self.device)
        attention_mask = encoding["attention_mask"].to(self.device)
        offset_mapping = encoding["offset_mapping"][0].tolist()

        with torch.no_grad():
            outputs = self.model(input_ids=input_ids, attention_mask=attention_mask)
            token_predictions = torch.argmax(outputs.logits, dim=2)[0].cpu().tolist()

        raw_spans = self.tokens_to_char_spans(
            token_predictions,
            offset_mapping,
            merge_distance=SPAN_MERGE_DISTANCE,
        )
        clean_spans = self.postprocess_spans(text, raw_spans, min_len=3, merge_distance=0)
        fragments = [text[s:e] for s, e in clean_spans]
        return SpanPrediction(
            spans=clean_spans,
            fragments=fragments,
            highlighted_text=self.highlight_text(text, clean_spans),
        )


# =========================
# Bot orchestration
# =========================
class ManipulationBotService:
    def __init__(self) -> None:
        self.device = DEVICE

    def analyze_text(self, text: str) -> str:
        text = text.strip()
        if not text:
            return "Надішли, будь ласка, текст повідомлення для аналізу."

        # 1. Спочатку спани
        span_service = SpanService(SPAN_MODEL_DIR, self.device)
        span_pred = span_service.predict_one(text)

        del span_service
        gc.collect()

        # 2. Потім техніки
        tech_service = TechniqueService(TECH_CHECKPOINT_PATH, self.device)
        tech_pred = tech_service.predict_one(text, max_labels=TECH_MAX_LABELS)

        del tech_service
        gc.collect()

        has_manipulation = bool(span_pred.spans or tech_pred.labels_uk)
        lines = [
            f"Маніпуляція: {'так' if has_manipulation else 'ні'}",
            "",
            "Техніки:",
        ]

        if tech_pred.labels_uk:
            for label in tech_pred.labels_uk:
                lines.append(f"• {label}")
        else:
            lines.append("• Не виявлено")

        lines.append("")
        lines.append("Маніпулятивні фрагменти:")
        if span_pred.fragments:
            for fragment in span_pred.fragments:
                lines.append(f"• {fragment}")
        else:
            lines.append("• Не виявлено")

        lines.append("")
        lines.append("Текст із підсвіткою:")
        lines.append(span_pred.highlighted_text if span_pred.spans else text)

        return "\n".join(lines)


def extract_message_text(update: Update) -> str:
    if update.message is None:
        return ""
    if update.message.text:
        return update.message.text
    if update.message.caption:
        return update.message.caption
    return ""


SERVICE: ManipulationBotService | None = None


async def start_handler(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    text = (
        "Привіт! Я бот для виявлення маніпуляцій у Telegram-постах.\n\n"
        "Надішли мені текст або перешли пост із каналу, і я поверну:\n"
        "• чи є маніпуляція;\n"
        "• які техніки виявлено;\n"
        "• де саме знаходяться маніпулятивні фрагменти."
    )
    if update.message is not None:
        await update.message.reply_text(text)


async def help_handler(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    text = (
        "Як користуватися:\n"
        "1. Надішли текст одним повідомленням або перешли пост.\n"
        "2. Отримай короткий аналіз.\n\n"
        "Команди:\n"
        "/start — запуск\n"
        "/help — довідка"
    )
    if update.message is not None:
        await update.message.reply_text(text)


SERVICE: ManipulationBotService | None = None


async def text_handler(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    global SERVICE
    text = extract_message_text(update)
    if not text:
        if update.message is not None:
            await update.message.reply_text(
                "Надішли текстове повідомлення або перешли пост із текстом / підписом."
            )
        return

    await context.bot.send_chat_action(chat_id=update.effective_chat.id, action=ChatAction.TYPING)

    try:
        if SERVICE is None:
            SERVICE = ManipulationBotService()

        result = SERVICE.analyze_text(text)

        if update.message is not None:
            await update.message.reply_text(result)

    except Exception as exc:
        logger.exception("Inference failed: %s", exc)
        if update.message is not None:
            await update.message.reply_text("Сталася помилка під час аналізу тексту.")


def validate_paths() -> None:
    if not SPAN_MODEL_DIR.exists():
        raise FileNotFoundError(f"Span model directory not found: {SPAN_MODEL_DIR}")
    if not TECH_CHECKPOINT_PATH.exists():
        raise FileNotFoundError(f"Technique checkpoint not found: {TECH_CHECKPOINT_PATH}")
    if not TELEGRAM_BOT_TOKEN:
        raise ValueError("Set TELEGRAM_BOT_TOKEN in environment variables or .env.")


def main() -> None:
    print("Starting bot...")
    print("BASE_DIR:", BASE_DIR)
    print("SPAN_MODEL_DIR exists:", SPAN_MODEL_DIR.exists(), SPAN_MODEL_DIR)
    print("TECH_CHECKPOINT_PATH exists:", TECH_CHECKPOINT_PATH.exists(), TECH_CHECKPOINT_PATH)
    print("TOKEN exists:", bool(TELEGRAM_BOT_TOKEN))

    validate_paths()

    application = Application.builder().token(TELEGRAM_BOT_TOKEN).build()
    application.add_handler(CommandHandler("start", start_handler))
    application.add_handler(CommandHandler("help", help_handler))
    application.add_handler(
        MessageHandler((filters.TEXT | filters.CAPTION) & ~filters.COMMAND, text_handler)
    )

    logger.info("Bot started. Device: %s", DEVICE)
    print("Bot started. Device:", DEVICE)
    application.run_polling(close_loop=False)


if __name__ == "__main__":
    main()
