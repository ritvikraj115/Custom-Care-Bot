import os
import json
import shutil
import faiss

BASE_DIR = "storage"
INDEX_ROOT = os.path.join(BASE_DIR, "indexes")
AUTOCOMPLETE_ROOT = os.path.join(BASE_DIR, "autocomplete")
SOCIAL_ROOT = os.path.join(BASE_DIR, "social")

def bot_dir(client_id, bot_id):
    return os.path.join(BASE_DIR, f"client_{client_id}", f"bot_{bot_id}")

def index_bot_dir(client_id, bot_id):
    return os.path.join(INDEX_ROOT, f"client_{client_id}", f"bot_{bot_id}")

def ensure_dir(path):
    os.makedirs(path, exist_ok=True)

def _delete_if_exists(path):
    if os.path.exists(path):
        shutil.rmtree(path)

def delete_bot_storage(client_id, bot_id):
    legacy_path = bot_dir(client_id, bot_id)
    index_path = index_bot_dir(client_id, bot_id)
    autocomplete_path = os.path.join(
        AUTOCOMPLETE_ROOT,
        f"client_{client_id}",
        f"bot_{bot_id}"
    )
    social_path = os.path.join(
        SOCIAL_ROOT,
        f"bot_{bot_id}.json"
    )

    _delete_if_exists(legacy_path)
    _delete_if_exists(index_path)
    _delete_if_exists(autocomplete_path)
    if os.path.exists(social_path):
        os.remove(social_path)
