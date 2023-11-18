from huggingface_hub import snapshot_download
model_id="michelcarroll/llama2-7b-earnings-stock-prediction-fine-tune"
snapshot_download(repo_id=model_id, local_dir="llama2-7b-earnings-stock-prediction-fine-tune", local_dir_use_symlinks=False, revision="main")