from huggingface_hub import snapshot_download

# The identifier for the scFoundation model on Hugging Face Hub
model_repo = "genbio-ai/scFoundation"
# The local directory where you want to save the model
local_model_dir = "checkpoints/scfoundation/scFoundation-model" 

print(f"Downloading model from {model_repo} to {local_model_dir}...")

# This function downloads all files from the repo to the specified local directory
snapshot_download(
    repo_id=model_repo,
    local_dir=local_model_dir,
    local_dir_use_symlinks=False  # Recommended to avoid symlink issues
)

print("Model downloaded successfully!")