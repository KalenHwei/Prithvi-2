from huggingface_hub import snapshot_download

DATASET_PATH = "/data6/personal/weiyongda/PhD/Weather/Prithvi-EO-2.0/dataset_for_temporal_corp/data"
repo_id = "ibm-nasa-geospatial/multi-temporal-crop-classification"
_ = snapshot_download(repo_id=repo_id, repo_type="dataset", cache_dir="/data6/personal/weiyongda/PhD/Weather/Prithvi-EO-2.0/dataset_for_temporal_corp", local_dir=DATASET_PATH, resume_download=True)