from huggingface_hub import HfApi

api = HfApi()

# Заміни на свої repo_id
SPAN_REPO = "yourname/manip-span-model"
TECH_REPO = "yourname/manip-tech-model"

api.create_repo(repo_id=SPAN_REPO, repo_type="model", private=True, exist_ok=True)
api.create_repo(repo_id=TECH_REPO, repo_type="model", private=True, exist_ok=True)

api.upload_folder(
    folder_path="models/spans",
    repo_id=SPAN_REPO,
    repo_type="model",
)

api.upload_folder(
    folder_path="models/tech",
    repo_id=TECH_REPO,
    repo_type="model",
)

print("Done")