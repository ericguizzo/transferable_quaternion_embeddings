import argparse
import os

import requests
from tqdm import tqdm

'''
Download our RH-emo pre-trained weights
Command line arguments define which task to download and where to put the checkpoint file.
'''


def download_file_from_google_drive(id, destination):
    URL = "https://docs.google.com/uc?export=download"

    session = requests.Session()

    response = session.get(URL, params = { 'id' : id }, stream = True)
    token = get_confirm_token(response)

    if token:
        params = { 'id' : id, 'confirm' : token }
        response = session.get(URL, params = params, stream = True)

    save_response_content(response, destination)


def get_confirm_token(response):
    for key, value in response.cookies.items():
        if key.startswith('download_warning'):
            return value

    return None


def save_response_content(response, destination):
    CHUNK_SIZE = 32768
    with open(destination, "wb") as f:
        with tqdm(unit='B', unit_scale=True, unit_divisor=1024) as bar:
            for chunk in response.iter_content(CHUNK_SIZE):
                if chunk:  # filter out keep-alive new chunks
                    f.write(chunk)
                    bar.update(CHUNK_SIZE)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--output_path', type=str, default="models")
    args = parser.parse_args()

    #file_id = '1vCX0KHW44Q9plKTdkgyKZRcyjfgVA7jX'
    file_id = '1gno2EONz2q9aPENkztIxcTkRdrXw3CkJ'
    if not os.path.exists(args.output_path):
        os.makedirs(args.output_path)

    output_path = os.path.join(args.output_path, "pretrained_RHemo")

    download_file_from_google_drive(file_id, output_path)

    print ('Pre-trained weights successfully downloaded')
