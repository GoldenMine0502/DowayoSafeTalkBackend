from datasets import load_dataset, tqdm


class DataConverterKmhas:
    def __init__(self):
        self.dataset = load_dataset("jeanlee/kmhas_korean_hate_speech")

        self.result_path = 'dataset/data/korean_kmhas_result.txt'

    def convert(self):
        # 0: Origin(출신차별) hate speech based on place of origin or identity;
        # 1: Physical(외모차별) hate speech based on physical appearance (e.g. body, face) or disability;
        # 2: Politics(정치성향차별) hate speech based on political stance;
        # 3: Profanity(혐오욕설) hate speech in the form of swearing, cursing, cussing, obscene words, or expletives; or an unspecified hate speech category;
        # 4: Age(연령차별) hate speech based on age;
        # 5: Gender(성차별) hate speech based on gender or sexual orientation (e.g. woman, homosexual);
        # 6: Race(인종차별) hate speech based on ethnicity;
        # 7: Religion(종교차별) hate speech based on religion;
        # 8: Not Hate Speech(해당사항없음).
        results = []
        for set_type in self.dataset:
            for value in tqdm(self.dataset[set_type]):
                result = '{}|{}'.format(value['text'], 1 if value['label'][0] != 8 else 0)
                results.append(result)

        with open(self.result_path, 'wt') as f:
            for result in results:
                f.write(result + '\n')


if __name__ == "__main__":
    DataConverterKmhas().convert()