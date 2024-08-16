import json
import random
import time
from pathlib import Path

from datasets import tqdm


class DataConverterDialect:
    def __init__(self):
        self.path = 'dataset/data/한국어 방언 발화(전라도)/Training/[라벨]전라도_학습데이터_1'
        self.path2 = 'dataset/data/한국어 방언 발화(경상도)/Training/[라벨]경상도_학습데이터_1'
        self.result_path = 'dataset/data/korean_dialect.txt'

    def clean_text(self, text):
        text = text.replace("(()) ", "").replace("(())", "")

        return text

    def convert(self):
        files = [f for f in Path(self.path).rglob('*.json') if f.is_file()]
        files.extend([f for f in Path(self.path2).rglob('*.json') if f.is_file()])

        total_list = []

        for file in tqdm(files):
            with open(file, 'rt', encoding='utf-8-sig') as f:
                jsondoc = json.load(f)

                utterances = jsondoc['utterance']
                texts = list(map(lambda utterance: self.clean_text(utterance['dialect_form']), utterances))
                # texts = 
                # print(texts)
                # time.sleep(1000)

                total_list.extend(texts)

        random.shuffle(total_list)

        count = 1000000  # len = 1992101 (전라) + 2088717 (경상)
        total_list = total_list[:count]

        with open(self.result_path, 'wt') as f:
            for text in tqdm(total_list):
                f.write(text)
                f.write('|0\n')


if __name__ == '__main__':
    jeolla = DataConverterDialect()
    jeolla.convert()