from pathlib import Path
import requests
from common.config import ROOT_PATH


def main():
    wiki_titles = ["北京市"]

    for title in wiki_titles:
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

        data_path = Path(ROOT_PATH) / "simple/data"
        if not data_path.exists():
            Path.mkdir(data_path)

        with open(data_path / f"{title}.txt", 'w') as fp:
            fp.write(wiki_text)


if __name__ == '__main__':
    main()
