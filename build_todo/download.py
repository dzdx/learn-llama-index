import os

import requests


def download(title: str, data_dir: str) -> str:
    response = requests.get(
        'https://zh.wikipedia.org/w/api.php',
        params={
            'action': 'query',
            'format': 'json',
            'titles': title,
            'prop': 'extracts',
            'explaintext': True,
        }
    ).json()
    page = next(iter(response['query']['pages'].values()))
    wiki_text = page['extract']

    if not os.path.exists(data_dir):
        os.makedirs(data_dir, exist_ok=True)

    file_path = os.path.join(data_dir, f"{title}")
    with open(file_path, 'w') as fp:
        fp.write(wiki_text)
    return file_path
