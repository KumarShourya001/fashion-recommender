
import os

BASE_DIR = "/content/drive/MyDrive/hm_fashion_project"
RAW_DATA_PATH = os.path.join(BASE_DIR, "data/raw")
BASE_PROCESS = "/content/drive/MyDrive/hm_fashion_project/data/processed"
IMAGE_PATH = os.path.join(BASE_PROCESS, "images_extracted")

PATHS = {
    "raw": RAW_DATA_PATH,
    "articles": os.path.join(BASE_PROCESS, "articles"),
    "customers": os.path.join(BASE_PROCESS, "customers"),
    "transactions": os.path.join(BASE_PROCESS, "transactions_train"),
    "images": os.path.join(RAW_DATA_PATH, "images"),
    "images_extracted": IMAGE_PATH,
    "embeddings": os.path.join(BASE_PROCESS, "embeddings")
}

def get_path(key):
    return PATHS.get(key, "Key not found")

def get_image_path(article_id):
    article_id_str = str(article_id).zfill(10)
    folder = article_id_str[:3]
    return os.path.join(IMAGE_PATH, folder, f"{article_id_str}.jpg")
