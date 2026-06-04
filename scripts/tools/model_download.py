# from huggingface_hub import snapshot_download

# # The identifier for the scFoundation model on Hugging Face Hub
# model_repo = "genbio-ai/scFoundation"
# # The local directory where you want to save the model
# local_model_dir = "checkpoints/scfoundation/scFoundation-model" 

# print(f"Downloading model from {model_repo} to {local_model_dir}...")

# # This function downloads all files from the repo to the specified local directory
# snapshot_download(
#     repo_id=model_repo,
#     local_dir=local_model_dir,
#     local_dir_use_symlinks=False  # Recommended to avoid symlink issues
# )

# print("Model downloaded successfully!")

from huggingface_hub import hf_hub_download
import os

if __name__ == "__main__":
    # specifyunder path
    target_dir = "/share/PertBench/checkpoints/CellFM/"
    os.makedirs(target_dir, exist_ok=True)

    # under file
    downloaded_path = hf_hub_download(
        repo_id="ShangguanNingyuan/CellFM",
        filename="CellFM_80M_weight.ckpt",
        cache_dir=target_dir
    )

    print(f"Model downloaded to: {downloaded_path}")
