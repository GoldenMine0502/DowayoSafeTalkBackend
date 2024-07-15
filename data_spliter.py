import random


class DataSpliter:
    def __init__(self):
        self.route = "dataset/result.txt"
        self.train = "dataset/train.txt"
        self.train_small = "dataset/train_small.txt"
        self.validation = "dataset/validation.txt"
        self.validation_small = "dataset/validation_small.txt"

    def split_data(self):
        ratio = 0.5
        all = 10000

        small = 0.05

        zero = int(round(all * ratio))
        one = all - zero

        zeros = []
        ones = []

        for line in open(self.route):
            text, label = line.split("|")

            if int(label) == 0:
                zeros.append(line)
            else:
                ones.append(line)

        trains = []
        trains_small = []
        validations = []
        validations_small = []

        zeros_small = int(round(len(zeros) * small))
        ones_small = int(round(len(ones) * small)) * 4

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