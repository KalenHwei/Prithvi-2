from huggingface_hub import snapshot_download

DATASET_PATH = "/data6/personal/weiyongda/PhD/Weather/Prithvi-EO-2.0/dataset_for_landslide/data"
repo_id = "ibm-nasa-geospatial/Landslide4sense"
_ = snapshot_download(repo_id=repo_id, repo_type="dataset", cache_dir="/data6/personal/weiyongda/PhD/Weather/Prithvi-EO-2.0/dataset_for_landslide/cache", local_dir=DATASET_PATH, resume_download=True)