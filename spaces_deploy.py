"""One-shot HF Space deploy: create the Space, upload the app files, set the API-key secret.

Requires `huggingface-cli login` (Write token) and an HF **PRO** subscription — as of
2026-07 hosting Gradio/Docker Spaces is PRO-only (Static is the only free type), and HF
removed the native Streamlit SDK, so this deploys as a Docker Space (see spaces_Dockerfile).
The free-tier demo lives on Streamlit Community Cloud instead. Run: python spaces_deploy.py
"""
from pathlib import Path
from huggingface_hub import HfApi

PROJ = Path(__file__).resolve().parent
SPACE_ID = "lilhuang15/ai-text-detector"
api = HfApi()

# 1. Create the Space (docker SDK — HF's supported way to run Streamlit now)
url = api.create_repo(SPACE_ID, repo_type="space", space_sdk="docker",
                      private=False, exist_ok=True)
print("space:", url, flush=True)

# 2. Upload the app files (renames applied: spaces_* -> canonical names)
api.upload_file(path_or_fileobj=PROJ / "app.py", path_in_repo="app.py",
                repo_id=SPACE_ID, repo_type="space")
api.upload_file(path_or_fileobj=PROJ / "spaces_requirements.txt", path_in_repo="requirements.txt",
                repo_id=SPACE_ID, repo_type="space")
api.upload_file(path_or_fileobj=PROJ / "spaces_README.md", path_in_repo="README.md",
                repo_id=SPACE_ID, repo_type="space")
api.upload_file(path_or_fileobj=PROJ / "spaces_Dockerfile", path_in_repo="Dockerfile",
                repo_id=SPACE_ID, repo_type="space")
api.upload_file(path_or_fileobj=PROJ / "src" / "__init__.py", path_in_repo="src/__init__.py",
                repo_id=SPACE_ID, repo_type="space")
api.upload_file(path_or_fileobj=PROJ / "src" / "claude_detector.py", path_in_repo="src/claude_detector.py",
                repo_id=SPACE_ID, repo_type="space")
print("files uploaded", flush=True)

# 3. Set the ANTHROPIC_API_KEY secret from the local .env (value never printed)
key = None
for line in (PROJ / ".env").read_text().splitlines():
    line = line.strip()
    if line.startswith("ANTHROPIC_API_KEY"):
        key = line.split("=", 1)[1].strip().strip('"').strip("'")
        break
assert key and key.startswith("sk-ant-"), "ANTHROPIC_API_KEY not found in .env"
api.add_space_secret(SPACE_ID, "ANTHROPIC_API_KEY", key)
print("secret set: ANTHROPIC_API_KEY (value hidden)", flush=True)

# 4. Report runtime/hardware
rt = api.get_space_runtime(SPACE_ID)
print("stage:", rt.stage, "| hardware:", rt.hardware or rt.requested_hardware or "cpu-basic (default)")
print("DONE")
