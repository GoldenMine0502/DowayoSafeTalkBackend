import os

import torch
import yaml
from datasets import tqdm
from matplotlib import pyplot as plt
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer, DebertaForSequenceClassification, DebertaConfig, AdamW

from yamlload import Config

device = "cuda" if torch.cuda.is_available() else "cpu"
os.environ["TOKENIZERS_PARALLELISM"] = "false"

def collate_fn(batch):
    data = []
    labels = []

    for text, label in batch:
        data.append(text)
        labels.append(label)

    # print(len(data), len(labels))
    # print(labels)

    return data, torch.tensor(labels)


class TextLoader(Dataset):
    def __init__(self, paths):
        self.data = []

        for path in paths:
            with open(path, 'rt') as file:
                for line in tqdm(file, ncols=50):
                    line = line.strip()
                    line = line.replace("/", " ")

                    self.data.append(line.split("|"))

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data[idx]
        return item[0], int(item[1])


class DebertaClassificationModel:
    def __init__(self, config):
        batch_size = config.train.batch_size
        num_workers = config.train.num_workers

        self.trainloader = DataLoader(dataset=TextLoader([config.data.train]),
                                      batch_size=batch_size,
                                      shuffle=True,
                                      num_workers=num_workers,
                                      collate_fn=collate_fn,
                                      pin_memory=True,
                                      drop_last=False,
                                      sampler=None)

        self.validationloader = DataLoader(dataset=TextLoader([config.data.validation]),
                                           batch_size=batch_size,
                                           shuffle=True,
                                           num_workers=num_workers,
                                           collate_fn=collate_fn,
                                           pin_memory=True,
                                           drop_last=False,
                                           sampler=None)

        self.testloader = None
        # model.config
        self.tokenizer = AutoTokenizer.from_pretrained("microsoft/deberta-base")
        model = DebertaForSequenceClassification.from_pretrained("microsoft/deberta-base")
        # model.config.max_position_embeddings = 1024
        # del model.config.id2label[1]

        # self.model = DebertaForSequenceClassification(model.config).to(device)
        # num_labels = len(model.config.id2label)

        model.config.num_labels = 2
        # model.config.max_position_embeddings = 768
        self.model = DebertaForSequenceClassification(model.config).to(device)
        # self.optimizer = torch.optim.AdamW(model.parameters(), lr=config.train.learning_rate)
        self.train_accuracy = []
        self.validation_accuracy = []

    def train_one(self, inputs, labels):
        inputs = self.tokenizer(inputs, return_tensors="pt", padding=True).to(device)
        labels = labels.to(device)

        # self.optimizer.zero_grad()
        output = self.model(**inputs, labels=labels)

        predicted_class_id = output.logits.argmax(dim=1)

        correct = 0
        for predict, ans in zip(predicted_class_id, labels):
            if predict == ans:
                correct += 1

        # self.optimizer.step()

        return output.loss.item(), correct, len(labels)

    def vali_one(self, inputs, labels):
        inputs = self.tokenizer(inputs, return_tensors="pt", padding=True).to(device)
        with torch.no_grad():
            logits = self.model(**inputs).logits

        predicted_class_id = logits.argmax(dim=1)
        # print(logits, predicted_class_id, labels)
        # print(logits.shape, labels.shape)

        correct = 0
        for predict, ans in zip(predicted_class_id, labels):
            if predict == ans:
                correct += 1

        return correct, len(labels)

    def train(self):
        self.model.train()

        losses = []
        corrects = 0
        total = 0
        for text, label in tqdm(self.trainloader, ncols=50):
            loss, correct, all = self.train_one(text, label)
            losses.append(loss)

            corrects += correct
            total += all

        train_accuracy = round(corrects / total * 100, 2)

        print("loss: {}, train accuracy: {}%".format(sum(losses) / len(losses), train_accuracy))

        return train_accuracy

    def validation(self):
        self.model.eval()

        counts = 0
        corrects = 0
        for text, label in tqdm(self.validationloader, ncols=50):
            correct, count = self.vali_one(text, label)

            corrects += correct
            counts += count

        validation_accuracy = round(corrects / counts * 100, 2)

        print("validation accuracy: {}%".format(validation_accuracy))

        return validation_accuracy

    def process(self, epoch=10, start_epoch=1):
        if start_epoch < 1:
            raise Exception("에포크는 1이상의 정수여야 합니다")

        if start_epoch > epoch:
            self.load_weights(start_epoch - 1)  # 현재 시작할 에포크 - 1당시 값으로 설정 후 학습

        for i in range(start_epoch, epoch + 1):
            print("epoch {}/{}:".format(i, epoch))
            self.train_accuracy.append(self.train())
            self.validation_accuracy.append(self.validation())

            torch.save(self.model, f'deberta_{i}.pt')

        self.show_plot(self.train_accuracy, self.validation_accuracy)

    def load_weights(self, epoch):
        self.train_accuracy.clear()
        self.validation_accuracy.clear()

        checkpoint = torch.load(f'chkpt/deberta_{epoch}.pth')

        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.train_accuracy.extend(checkpoint['train_accuracy'])
        self.validation_accuracy.extend(checkpoint['validation_accuracy'])

    def save_weights(self, epoch, train_acc, validation_acc):
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'train_accuracy': train_acc,
            'validation_accuracy': validation_acc
        }

        os.makedirs('chkpt', exist_ok=True)
        torch.save(checkpoint, f'chkpt/deberta_{epoch}.pth')

    def show_plot(self, train_accuracies, validation_accuracies):
        # 에포크 수
        epochs = range(1, len(train_accuracies) + 1)

        # 플롯 크기 설정
        plt.figure(figsize=(10, 6))

        # train 정확도 플로팅
        plt.plot(epochs, train_accuracies, 'b', label='Train Accuracy')

        # validation 정확도 플로팅
        plt.plot(epochs, validation_accuracies, 'r', label='Validation Accuracy')

        # 제목 및 축 레이블 설정
        plt.title('Train and Validation Accuracy')
        plt.xlabel('Epochs')
        plt.ylabel('Accuracy')

        # 범례 추가
        plt.legend()

        # 그래프 보여주기
        plt.show()


# 학습 안시키면 정확도 51%
if __name__ == "__main__":
    config = Config('config/config.yml')

    print('text loader is loaded')

    trainer = DebertaClassificationModel(config)
    trainer.process(
        epoch=config.train.epoch,
        start_epoch=config.train.start_epoch
    )

# pip3 freeze > requirements.txt
# pip install -r requirements.txt
