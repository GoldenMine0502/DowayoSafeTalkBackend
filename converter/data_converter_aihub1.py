import re
from pathlib import Path

from datasets import tqdm

# 한글 범위의 정규 표현식 패턴
korean_pattern = re.compile('[^ ㄱ-ㅣ가-힣]+')


def extract_korean(text):
    # 정규 표현식을 사용하여 한글만 추출
    korean_only = korean_pattern.sub('', text)

    return korean_only
class DataConverterAihub1:
    def __init__(self):
        self.path = 'D:/datasets/한국어 음성/한국어_음성_분야'
        pass

    def convert(self):
        # 개수 확인결과 622545개
        all_files = [f for f in Path(self.path).rglob('*.txt') if f.is_file()]

        result_file = 'dataset/korean_aihub1_result.txt'

        os = open(result_file, 'wt', encoding='utf-8')

        for path in tqdm(all_files, ncols=60):
            with open(path, 'rt') as file:
                text = file.read().strip().replace("  ", " ")

            text = extract_korean(text).strip(' ')
            os.write(f"{text}|0\n")

        os.close()

if __name__ == '__main__':
    aihub1 = DataConverterAihub1()
    aihub1.convert()

