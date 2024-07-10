import random


class DataSpliter:
    def __init__(self):
        self.route = "dataset/all.txt"
        self.train = "dataset/train.txt"
        self.validation = "dataset/validation.txt"

    def split_data(self):
        ratio = 0.5
        all = 2000

        zero = int(round(all * ratio))
        one = all - zero

        zeros = []
        ones = []

        for line in open(self.route):
            text, label = line.split("|")

            if label == 0:
                zeros.append(line)
            else:
                ones.append(line)

        trains = []
        validations = []

        # 비율에 맞게 넣기
        validations.extend(zeros[0:zero])
        validations.extend(ones[0:one])

        # 나머지 데이터는 학습셋
        trains.extend(zeros[zero:])
        trains.extend(ones[one:])

        with open("train.txt", "wt") as file:
            for train in trains:
                file.write(train)
                file.write('\n')

        with open("validation.txt", "wt") as file:
            for validation in validations:
                file.write(validation)
                file.write('\n')


if __name__ == "__main__":
    spliter = DataSpliter()
    spliter.split_data()