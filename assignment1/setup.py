import os
import urllib.request
import zipfile
import argparse

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--emb", type=str, default="https://dl.fbaipublicfiles.com/fasttext/vectors-english/crawl-300d-2M.vec.zip")
    args = parser.parse_args()
    return args


def download_and_extract(url, extract_path):
    if not os.path.exists(extract_path):
        os.makedirs(extract_path)

    zip_file_path = os.path.join(extract_path, 'crawl-300d-2M.vec.zip')
    vec_file_path = os.path.join(extract_path, 'crawl-300d-2M.vec')

    if os.path.isfile(vec_file_path):
        print(f"{vec_file_path} exists")
    elif os.path.isfile(zip_file_path):
        print(f"{zip_file_path} exists. Unzipping...")
        with zipfile.ZipFile(zip_file_path, 'r') as zip_ref:
            zip_ref.extractall(extract_path)
        print(f"Extraction complete. Embeddings saved to {extract_path}")
    else:
        print(f"Downloading embeddings from {url}...")
        urllib.request.urlretrieve(url, zip_file_path)
        print(f"Downloaded to {zip_file_path}")
        
        print(f"Unzipping {zip_file_path}...")
        with zipfile.ZipFile(zip_file_path, 'r') as zip_ref:
            zip_ref.extractall(extract_path)
        print(f"Extraction complete. Embeddings saved to {extract_path}")


def main():
    args = get_args()
    url = args.emb
    extract_path = os.path.join(os.getcwd(), 'pre_trained_embeddings')

    download_and_extract(url, extract_path)

if __name__ == "__main__":
    main()
