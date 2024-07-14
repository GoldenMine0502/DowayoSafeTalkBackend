import pandas as pd
from datasets import tqdm


class DataConverterSmilegateUnsmile:
    def __init__(self):
        self.path = "dataset/data/lgbt/20200402_koco_merged_test.tsv"
        self.result_path = "dataset/data/korean_lgbt_result.txt"

    def convert(self):
        # ,문장,혐오 여부
        # 0,지금 어디 계세요?,0
        csv = pd.read_csv(self.path, sep='\t')

        results = []

        # contents	label
        # 기자들 이제 게이기삿거리 찾아다니겠네 ㅋㅋㅋ반응이월드컵	1.0
        for index in tqdm(range(0, csv.shape[0])):
            row = csv.iloc[index]
            text = row['contents'].replace('\n', ' ')
            is_badword = int(float(row['label']))

            final_text = '{}|{}'.format(text, 1 if is_badword else 0)
            results.append(final_text)

        file = open(self.result_path, "wt")

        for result in results:
            file.write(result + "\n")

        file.close()


if __name__ == "__main__":
    converter = DataConverterSmilegateUnsmile()
    converter.convert()