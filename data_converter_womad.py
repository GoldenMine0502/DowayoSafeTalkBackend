import pandas as pd
from datasets import tqdm


class DataConverterWomad:
    def __init__(self):
        self.path = "dataset/data/extremist_website_womad/hate_speech_data.csv"
        self.result_path = "dataset/data/korean_womad_result.txt"

    def convert(self):
        # ,문장,혐오 여부
        # 0,지금 어디 계세요?,0
        csv = pd.read_csv(self.path)

        results = []

        for index in tqdm(range(0, csv.shape[0])):
            row = csv.iloc[index]
            text = row['문장'].replace('\n', ' ')
            is_badword = int(row['혐오 여부'])

            final_text = '{}|{}'.format(text, 1 if is_badword else 0)
            results.append(final_text)

        file = open(self.result_path, "wt")

        for result in results:
            file.write(result + "\n")

        file.close()


if __name__ == "__main__":
    converter = DataConverterWomad()
    converter.convert()