import random

from datasets import tqdm


class DataSpliter:
    def __init__(self):
        self.route = "dataset/result.txt"
        self.train = "dataset/train.txt"
        self.train_small = "dataset/train_small.txt"
        self.validation = "dataset/validation.txt"
        self.validation_small = "dataset/validation_small.txt"

    def distinct(self, lst):
        seen = set()
        result = []
        for item in tqdm(lst, ncols=80):
            if item not in seen:
                seen.add(item)
                result.append(item)
        return result

    def split_data(self):
        maxlen = 250
        minlen = 0  # 단순 한글자 욕설 떄문에 냅둠
        ratio = 0.5
        all = 15000

        small = 0.25

        zero = int(round(all * ratio))
        one = all - zero

        zeros = []
        ones = []

        for line in open(self.route):
            text, label = line.split("|")

            if len(text) > maxlen:
                continue

            if len(text) < minlen:
                continue

            if int(label) == 0:
                zeros.append(line)
            else:
                ones.append(line)

        zeros = self.distinct(zeros)
        ones = self.distinct(ones)

        random.shuffle(zeros)
        random.shuffle(ones)

        trains = []
        trains_small = []
        validations = []
        validations_small = []

        zeros_small = int(round(len(zeros) * small))
        ones_small = int(round(len(ones) * small))

        # 비율에 맞게 넣기
        validations.extend(zeros[0:zero])
        validations.extend(ones[0:one])
        validations_small.extend(zeros[:zeros_small][0:zero])
        validations_small.extend(ones[:ones_small][0:one])

        # 나머지 데이터는 학습셋
        trains.extend(zeros[zero:])
        trains.extend(ones[one:])
        trains_small.extend(zeros[:zeros_small][zero:])
        trains_small.extend(ones[:ones_small][one:])

        print(len(zeros), len(ones))
        print(len(trains), len(validations))

        trains.sort(key=len, reverse=True)
        validations.sort(key=len, reverse=True)

        with open(self.train, "wt") as file:
            for train in trains:
                file.write(train)
                # file.write('\n')

        with open(self.validation, "wt") as file:
            for validation in validations:
                file.write(validation)
                # file.write('\n')

        with open(self.train_small, "wt") as file:
            for train in trains_small:
                file.write(train)
                # file.write('\n')

        with open(self.validation_small, "wt") as file:
            for validation in validations_small:
                file.write(validation)
                # file.write('\n')


if __name__ == "__main__":
    spliter = DataSpliter()
    spliter.split_data()