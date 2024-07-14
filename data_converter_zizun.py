from datasets import tqdm


class DataConverterZizun:
    def __init__(self):
        self.paths = [
            "dataset/data/korean-malicious-comments-dataset/Dataset.csv",
        ]
        self.result_path = "dataset/data/korean_zizun_result.txt"

    def convert(self):
        lines = []

        # 문장	여성/가족	남성	성소수자	인종/국적	연령	지역	종교	기타 혐오	악플/욕설	clean	개인지칭
        # 일안하는 시간은 쉬고싶어서 그런게 아닐까	0	0	0	0	0	0	0	0	0	1	0
        for path in self.paths:
            with open(path, "rt") as f:
                lines.extend(map(lambda x: x.strip(), f.readlines()[1:]))

        results = []

        #  content	lable
        # 이종석 한효주 나오는 드라마 이후로 드라마 안봤다. 2년전인가?? 좀 신선했었지. 근데 이런 개막장 드라마는 도대체 누가 보느냐면 변태들이 보는 것이다. 정상적인 사람들은 채널을 돌리게 된다.	0
        for line in tqdm(lines):
            text, clean = line.split('\t')

            is_badword = '0' in clean
            final_text = '{}|{}'.format(text, 1 if is_badword else 0)
            results.append(final_text)

        file = open(self.result_path, "wt")

        for result in results:
            file.write(result + "\n")

        file.close()


if __name__ == "__main__":
    converter = DataConverterZizun()
    converter.convert()