import time

from PyKomoran import *
from datasets import tqdm


class PreProcessKomoran:
    def __init__(self):
        # print(__version__)
        self.komoran = Komoran("STABLE")  # OR EXP
        self.datasets = [
            './dataset/data/korean_aihub1_result.txt',
            './dataset/data/korean_selectstar_result.txt',
        ]

        self.result_path = 'dataset/train.txt'

    def clean_text(self, text):
        # 즵 즺 즫 즥 즷 즴 즨 즹 즬 즿 즼 즽 즻 즻 즾
        return (text.replace("#@이름#", "#")
                    .replace(" # ", "#")
                    .replace("##", "#")
                    .replace('즺', '짖')
                    .replace('즵', '집')
                    .replace('즫', '짇')
                    .replace('즥', '직')
                    .replace('즷', '짓')
                    .replace('즴', '짐')
                    .replace('즨', '진')
                    .replace('즹', '징')
                    .replace('즬', '질')
                    .replace('즿', '짛')
                    .replace('즼', '짘')
                    .replace('즽', '짙')
                    .replace('즻', '짗')
                    .replace('즾', '짚')
                    .replace('즤', '지')
                )


    def get_tokens(self):
        all_texts = []

        results = []

        for dataset in self.datasets:
            for line in open(dataset):
                all_texts.append(line.strip())

        for text in tqdm(all_texts):
            def filter_sw(text):
                split = text.split("/")

                if len(split) == 1:
                    return '#' not in text

                return split[1] != 'SW'
            # print(text)
            text = self.clean_text(text)
            text, label = text.split("|")

            res = self.komoran.get_plain_text(text).split(' ')

            res = list(filter(filter_sw, res))  # 특수문자 필터링
            res = list(map(lambda x: x.split('/')[0], res))  # 대한민국/NNP 같은 단어가 있으면 슬래시 뒤 문자 떼버림
            # print(text, res, len(res))
            results.append((res, label))

            # time.sleep(1)

        os = open(self.result_path, 'wt', encoding='utf-8')
        for all_text, label in tqdm(results):
            for i in range(len(all_text)):
                os.write(all_text[i])
                if i != len(all_text) - 1:
                    os.write('/')

            os.write('|')
            os.write(str(label))
            os.write('\n')

        os.close()

        # validation all texts are writed normally
        with open(self.result_path, 'rt', encoding='utf-8') as file:
            lines = len(file.readlines())
            print('lines:', lines)

            assert lines == len(results)


if __name__ == '__main__':
    # komoran = Komoran("EXP")  # OR EXP
    # print(komoran.get_plain_text("대한민국은  민주 공화  국이다.").split(' '))
    pre_komoran = PreProcessKomoran()
    pre_komoran.get_tokens()