from datasets import tqdm


class DataConverterKocohub:
    def __init__(self):
        self.paths = [
            "dataset/data/korean-hate-speech/labeled/train.tsv",
            "dataset/data/korean-hate-speech/labeled/dev.tsv",
        ]
        self.result_path = "dataset/data/korean_kocohub_result.txt"

    def convert(self):
        lines = []

        for path in self.paths:
            with open(path, "rt") as f:
                lines.extend(map(lambda x: x.strip(), f.readlines()[1:]))

        results = []

        for line in tqdm(lines):
            text, gender_bias, bias, hate = line.split('\t')

            is_badword = gender_bias == 'True' or hate != 'none' or bias != 'none'

            final_text = '{}|{}'.format(text, 1 if is_badword else 0)
            results.append(final_text)

        file = open(self.result_path, "wt")

        for result in results:
            file.write(result + "\n")

        file.close()


if __name__ == "__main__":
    converter = DataConverterKocohub()
    converter.convert()