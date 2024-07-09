

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
from transformers import AutoTokenizer, DebertaForSequenceClassification

device = "cuda" if torch.cuda.is_available() else "cpu"

class TextLoader:
    def __init__(self, paths):
        self.data = []

        for path in paths:
            with open(path, 'rt') as file:
                for line in file:
                    line = line.strip()
                    split = line.split('|')
                    # print(split)
                    self.data.append((split[0], int(split[1])))


class DebertaClassificationModel:
    def __init__(self, trainloader, validationloader, testloader, checkpoint=None):
        self.trainloader = trainloader
        self.validationloader = validationloader
        self.testloader = testloader

        # model.config
        self.tokenizer = AutoTokenizer.from_pretrained("microsoft/deberta-base")
        model = DebertaForSequenceClassification.from_pretrained("microsoft/deberta-base").to(device)
        # model.config.max_position_embeddings = 1024
        # del model.config.id2label[1]

        # self.model = DebertaForSequenceClassification(model.config).to(device)
        self.model = DebertaForSequenceClassification.from_pretrained("microsoft/deberta-base", num_labels=1).to(device)

        if checkpoint is not None:
            self.model = torch.load(checkpoint).to(device)

    def train_one(self, text, label):
        text = text.replace("/", " ")
        inputs = self.tokenizer(text, return_tensors="pt").to(device)

        # To train a model on `num_labels` classes, you can pass `num_labels=num_labels` to `.from_pretrained(...)`
        # labels = torch.tensor([label], dtype=torch.int32).to(device)
        labels = torch.tensor([label]).to(device)
        loss = self.model(**inputs, labels=labels).loss

        return loss.item()

    def vali_one(self, text, label):
        text = text.replace("/", " ")
        inputs = self.tokenizer(text, return_tensors="pt").to(device)

        with torch.no_grad():
            logits = self.model(**inputs).logits

        predicted_class_id = logits.argmax().item()
        print(predicted_class_id)

        return predicted_class_id == label

    def train(self):
        losses = []
        for text, label in tqdm(self.trainloader.data, ncols=50):
            loss = self.train_one(text, label)
            losses.append(loss)

        print("loss: {}".format(sum(losses) / len(losses)))

    def validation(self):
        count = 0
        correct = 0
        for text, label in tqdm(self.validationloader.data, ncols=50):
            state = self.vali_one(text, label)

            count += 1
            if state:
                correct += 1

        print("validation accuracy: {}%".format(round(correct / count * 100, 2)))


    def process(self, epoch=10):
        for i in range(1, epoch + 1):
            print("epoch {}:".format(i))
            self.train()
            self.validation()


if __name__ == "__main__":
    l1 = TextLoader(["./dataset/train.txt"])
    l2 = TextLoader(["./dataset/validation.txt"])
    # testloader = TextLoader(["./dataset/validation.txt"]
    l3 = None  # TODO

    print('text loader is loaded')

    trainer = DebertaClassificationModel(l1, l2, l3)
    trainer.process(epoch=10)

    torch.save(trainer.model, 'deberta.pt')
