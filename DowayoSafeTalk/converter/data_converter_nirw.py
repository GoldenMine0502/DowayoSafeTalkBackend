import json
import random
from pathlib import Path

from datasets import tqdm


class DataConverterNirw:
    def __init__(self):
        self.path = 'dataset/data/NIKL_NEWSPAPER_2023_v1.0'
        self.files = [f for f in Path(self.path).rglob('*.json') if f.is_file()]
        self.result_path = 'dataset/data/korean_nirw_result.txt'

    def convert(self):
        lines = []
        lines_politics = []

        #   "document": [
        #     {
        #       "id": "NIRW2300000001.1",
        #       "metadata": {
        #         "title": "노컷뉴스 2022년 기사",
        #         "author": "박중석 부산CBS 박중석 기자",
        #         "publisher": "노컷뉴스",
        #         "date": "20220101",
        #         "topic": "정치",
        #         "original_topic": "지역>부산, 지역>경남, 지역>울산"
        #       },
        #       "paragraph": [
        #         {
        #           "id": "NIRW2300000001.1.1",
        #           "form": "[영상]“위기를 이겨내고 일상으로” 부산 기관장 신년사"
        #         },
        #         {
        for file in tqdm(self.files):
            with open(file, 'rt') as f:
                jsondoc = json.load(f)
                # print(jsondoc['document'][0])
                for doc in jsondoc['document']:
                    topic = doc['metadata']['topic']

                    texts = map(lambda x: x['form'].replace('\r\n', ' ').replace('\r', ' ').replace('\n', ' '), doc['paragraph'])

                    if topic == '정치':
                        lines_politics.extend(texts)
                    else:
                        lines.extend(texts)
        # print(lines)
        print(len(lines), len(lines_politics))

        # 각분야에 대해 특정 개수만큼 추출
        total_count = 250_0000
        total_count_politics = 50_0000
        random.shuffle(lines)
        random.shuffle(lines_politics)

        lines = lines[:total_count]
        lines_politics = lines_politics[:total_count_politics]

        with open(self.result_path, 'wt') as f:
            for line in tqdm(lines):
                f.write('{}|{}\n'.format(line, 0))
            for line in tqdm(lines_politics):
                f.write('{}|{}\n'.format(line, 0))

        with open(self.result_path, 'rt') as f:
            assert len(f.readlines()) == total_count + total_count_politics


if __name__ == '__main__':
    converter = DataConverterNirw()
    converter.convert()