"""Скачивает и распаковывает датасет Kvasir-SEG в ./data/Kvasir-SEG.

    python download_data.py                    # скачать официальный архив
    python download_data.py --zip archive.zip  # распаковать уже скачанный архив
                                               # (официальный или Kaggle-зеркало)
"""
import argparse
import os
import shutil
import sys
import tempfile
import zipfile

import requests
from tqdm import tqdm

import config


def download(url, dst):
    with requests.get(url, stream=True, timeout=60) as r:
        r.raise_for_status()
        total = int(r.headers.get("content-length", 0))
        with open(dst, "wb") as f, tqdm(total=total, unit="B", unit_scale=True, desc="Kvasir-SEG") as bar:
            for chunk in r.iter_content(chunk_size=1 << 20):
                f.write(chunk)
                bar.update(len(chunk))


def find_dataset_root(directory):
    """Ищет папку, в которой лежат images/ и masks/ (структура архивов бывает разной)."""
    for root, dirs, _ in os.walk(directory):
        if "images" in dirs and "masks" in dirs:
            return root
    return None


def is_ready(data_dir):
    return all(
        os.path.isdir(os.path.join(data_dir, d)) and os.listdir(os.path.join(data_dir, d))
        for d in ("images", "masks")
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--zip", help="путь к уже скачанному zip-архиву")
    parser.add_argument("--url", default=config.DATASET_URL, help="откуда скачивать архив")
    args = parser.parse_args()

    target = config.DATA_DIR
    if is_ready(target):
        print(f"Датасет уже на месте: {target}")
        return

    os.makedirs(os.path.dirname(target), exist_ok=True)
    with tempfile.TemporaryDirectory(dir=os.path.dirname(target)) as tmp:
        archive = args.zip
        if archive is None:
            archive = os.path.join(tmp, "kvasir-seg.zip")
            print(f"Скачиваю {args.url}")
            try:
                download(args.url, archive)
            except requests.RequestException as e:
                sys.exit(
                    f"Не удалось скачать архив: {e}\n"
                    f"Скачайте его вручную (официальный сайт https://datasets.simula.no/kvasir-seg/ "
                    f"или Kaggle) и запустите: python download_data.py --zip <путь к zip>"
                )

        print("Распаковываю...")
        extract_dir = os.path.join(tmp, "extracted")
        with zipfile.ZipFile(archive) as zf:
            zf.extractall(extract_dir)

        root = find_dataset_root(extract_dir)
        if root is None:
            sys.exit("В архиве не найдены папки images/ и masks/")

        if os.path.exists(target):
            shutil.rmtree(target)
        os.makedirs(target)
        for d in ("images", "masks"):
            shutil.move(os.path.join(root, d), os.path.join(target, d))

    n_img = len(os.listdir(os.path.join(target, "images")))
    n_msk = len(os.listdir(os.path.join(target, "masks")))
    print(f"Готово: {target} (images: {n_img}, masks: {n_msk})")


if __name__ == "__main__":
    main()
