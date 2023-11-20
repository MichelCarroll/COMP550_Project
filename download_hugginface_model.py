from huggingface_hub import snapshot_download
model_id="llama2-7b-earnings-stock-prediction-fine-tune-1000-examples-binary-v2"
snapshot_download(repo_id=f"michelcarroll/{model_id}", local_dir=model_id, local_dir_use_symlinks=False, revision="main")