from datasets import tqdm


class DataConverterSmilegateUnsmile:
    def __init__(self):
        self.paths = [
            "dataset/data/korean_unsmile_dataset/unsmile_train_v1.0.tsv",
            "dataset/data/korean_unsmile_dataset/unsmile_valid_v1.0.tsv",
        ]
        self.result_path = "dataset/data/korean_smilegate_result.txt"

    def convert(self):
        lines = []

        # 문장	여성/가족	남성	성소수자	인종/국적	연령	지역	종교	기타 혐오	악플/욕설	clean	개인지칭
        # 일안하는 시간은 쉬고싶어서 그런게 아닐까	0	0	0	0	0	0	0	0	0	1	0
        for path in self.paths:
            with open(path, "rt") as f:
                lines.extend(map(lambda x: x.strip(), f.readlines()[1:]))

        results = []

        for line in tqdm(lines):
            text, gender_f, gender_m, sexual, nation, age, local, religion, others, bad_word, clean, personal = line.split('\t')

            is_badword = not int(clean)

            final_text = '{}|{}'.format(text, 1 if is_badword else 0)
            results.append(final_text)

        file = open(self.result_path, "wt")

        for result in results:
            file.write(result + "\n")

        file.close()


if __name__ == "__main__":
    converter = DataConverterSmilegateUnsmile()
    converter.convert()