# from transformers import DebertaConfig, DebertaModel
# from transformers import DebertaTokenizer
#
# # Initializing a DeBERTa microsoft/deberta-base style configuration
# configuration = DebertaConfig()
#
# # Initializing a model (with random weights) from the microsoft/deberta-base style configuration
# model = DebertaModel(configuration)
#
# # Accessing the model configuration
# configuration = model.config
#
#
# tokenizer = DebertaTokenizer.from_pretrained("microsoft/deberta-base")
# tokenizer("Hello world")["input_ids"]
# [1, 31414, 232, 2]
#
# tokenizer(" Hello world")["input_ids"]
# [1, 20920, 232, 2]

# from transformers import AutoTokenizer, DebertaModel
#
# tokenizer = AutoTokenizer.from_pretrained("microsoft/deberta-base")
# model = DebertaModel.from_pretrained("microsoft/deberta-base")
#
# # Hello [1, 3, 768]
# # Hello, World! [1, 6, 768]
# # a [1, 3, 768]
# # apple is nice [1, 5, 768] [[    1, 27326,    16,  2579,     2]]
# # banana is nice [1, 6, 8] tensor([[   1, 7384, 1113,   16, 2579,    2]]
#
# inputs = tokenizer("apple is nice", return_tensors="pt")
# print(inputs)
# outputs = model(**inputs)
#
# last_hidden_states = outputs.last_hidden_state
#
# print(last_hidden_states, last_hidden_states.shape)

import torch
from datasets import tqdm
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer, DebertaForSequenceClassification, DebertaConfig

device = "cuda" if torch.cuda.is_available() else "cpu"


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
    def __init__(self, trainloader, validationloader, testloader, checkpoint=None):
        self.trainloader = trainloader
        self.validationloader = validationloader
        self.testloader = testloader

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

        if checkpoint is not None:
            self.model = torch.load(checkpoint).to(device)

    def train_one(self, inputs, labels):
        inputs = self.tokenizer(inputs, return_tensors="pt", padding=True).to(device)
        labels = labels.to(device)

        loss = self.model(**inputs, labels=labels).loss

        return loss.item()

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
        losses = []
        for text, label in tqdm(self.trainloader, ncols=50):
            loss = self.train_one(text, label)
            losses.append(loss)

        print("loss: {}".format(sum(losses) / len(losses)))

    def validation(self):
        counts = 0
        corrects = 0
        for text, label in tqdm(self.validationloader, ncols=50):
            correct, count = self.vali_one(text, label)

            corrects += correct
            counts += count

        print("validation accuracy: {}%".format(round(corrects / counts * 100, 2)))

    def process(self, epoch=10):
        for i in range(1, epoch + 1):
            print("epoch {}/{}:".format(i, epoch))
            self.train()
            self.validation()

            torch.save(self.model, f'deberta_{i}.pt')


# 학습 안시키면 정확도 51%
if __name__ == "__main__":
    batch_size = 2
    # l1 = TextLoader(["./dataset/train.txt"])
    # l2 = TextLoader(["./dataset/validation.txt"])
    # testloader = TextLoader(["./dataset/validation.txt"]
    # l3 = None  # TODO
    l1 = DataLoader(dataset=TextLoader(["./dataset/train.txt"]),
                    batch_size=batch_size,
                    shuffle=True,
                    num_workers=1,
                    collate_fn=collate_fn,
                    pin_memory=True,
                    drop_last=False,
                    sampler=None)

    l2 = DataLoader(dataset=TextLoader(["./dataset/validation.txt"]),
                    batch_size=batch_size,
                    shuffle=True,
                    num_workers=1,
                    collate_fn=collate_fn,
                    pin_memory=True,
                    drop_last=False,
                    sampler=None)

    l3 = None

    print('text loader is loaded')

    trainer = DebertaClassificationModel(l1, l2, l3)
    trainer.process(epoch=20)

# pip3 freeze > requirements.txt
# pip install -r requirements.txt