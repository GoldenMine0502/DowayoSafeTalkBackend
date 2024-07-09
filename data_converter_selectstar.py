import json
from pathlib import Path

from tqdm import tqdm


class DataConverter:
    pass


# 데이터셋은 결국 text|value로 회귀한다.
class DataConverterSelectStar:
    def __init__(self):
        self.path = "dataset/Selectstar_Tunip_HUMANE Lab_opendata"
        pass

    def convert_all(self):
        all_files = [f for f in Path(self.path).iterdir() if f.is_file()]
        bad_content_keywords = [
            "모욕", "욕설", "외설", "폭력위협/범죄조장", "성혐오",
            "연령", "인종/지역", "장애", "종교", "정치성향", "직업"
        ]

        results = []

        zeros = 0

        for file_path in tqdm(all_files, ncols=50):
            with open(file_path, "rt", encoding="utf-8-sig") as file:
                # {"모욕": 1, "욕설": 0, "외설": 0, "폭력위협/범죄조장": 0, "성혐오": 0, "연령": 0, "인종/지역": 0,
                # "장애": 0, "종교": 0, "정치성향": 1, "직업": 0, "1단계Y/N": "Y",
                # "2-1단계Y/N": "Y", "2-2단계Y/N": "N",
                # "문장": "독점은 당연히 막아야지. 이건 사장경제는 경쟁으로 가야 정상순환되는데
                # 독점이 되면 기업이 공산국가처럼 독재하는건데",
                # "대상하이라이트": "독점은 당연히 막아야지. 이건 사장경제는 경쟁으로 가야 정상순환되는데 독점이 되면 기업이 ''공산국가'''처럼 독재하는건데",
                # "정치성향Y/N": "Y", "혐오 클래스": "Y", "특정 집단": "N", "그 외": "N"}
                file_text = file.read().strip().replace("\n", "").replace("'", "").replace('\'', '')
                data = json.loads(file_text)

                text = data["문장"].strip()

                if len(text) < 3:
                    continue

                bad_content_sum = sum(map(lambda x: data[x], bad_content_keywords))

                score = 1 if bad_content_sum > 0 else 0
                results.append(f"{text}|{score}")

                if score == 0:
                    zeros += 1

        path = "dataset/selectstar_result.txt"

        with open(path, "w") as file:
            file.write("\n".join(results))

        print('zeros:', zeros)

if __name__ == "__main__":
    converter = DataConverterSelectStar()
    converter.convert_all()

