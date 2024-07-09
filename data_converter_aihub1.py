from pathlib import Path

from datasets import tqdm

from text_util import extract_korean


class DataConverterAihub1:
    def __init__(self):
        self.path = 'D:/datasets/한국어 음성/한국어_음성_분야'
        pass

    def convert(self):
        # 개수 확인결과 622545개
        all_files = [f for f in Path(self.path).rglob('*.txt') if f.is_file()]

        result_file = 'datasets/korean_aihub1_result.txt'

        os = open(result_file, 'wt', encoding='utf-8')

        for path in tqdm(all_files, ncols=60):
            with open(path, 'rt') as file:
                text = file.read().strip()

            text = extract_korean(text).strip(' ')
            os.write(f"{text}|0\n")

        os.close()

if __name__ == '__main__':
    aihub1 = DataConverterAihub1()
    aihub1.convert()