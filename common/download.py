import os

import requests

wiki_titles = ["北京市", "上海市", "杭州市", "广州市", "南京市"]


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

    file_path = os.path.join(data_dir, f"{title}.txt")
    with open(file_path, 'w') as fp:
        fp.write(wiki_text)
    return file_path
