import os

import torch
from datasets import tqdm
from kobert_tokenizer import KoBERTTokenizer
from matplotlib import pyplot as plt
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torchsummary import summary
from transformers import AutoTokenizer, DebertaV2ForSequenceClassification, DebertaV2Config, pipeline, \
    DebertaForSequenceClassification

from yamlload import Config

device = "cuda" if torch.cuda.is_available() else "cpu"
os.environ["TOKENIZERS_PARALLELISM"] = "false"


def collate_fn(batch):
    data = []
    labels = []

    # max_len = 0

    for text, label in batch:
        data.append(text)
        labels.append([1.0 if label == 0 else 0.0, 1.0 if label == 1 else 0.0])
        # max_len = max(max_len, len(text))

    # data_pad = []
    # for text in data:
    #     data_pad.append(text + " " * (max_len - len(text)))
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
    def __init__(self, config, only_inference=False):
        batch_size = config.train.batch_size
        num_workers = config.train.num_workers

        if not only_inference:
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

        # model.config skt/kobert-base-v1
        self.tokenizer = KoBERTTokenizer.from_pretrained("skt/kobert-base-v1")
        model = DebertaV2ForSequenceClassification.from_pretrained("microsoft/deberta-v3-large")
        # # model.config.max_position_embeddings = 1024
        # # del model.config.id2label[1]
        #
        # # self.model = DebertaForSequenceClassification(model.config).to(device)
        # # num_labels = len(model.config.id2label)
        #
        # model.config.num_labels = 2
        # model.config.hidden_dropout_prob = 0.01
        # model.config.attention_probs_dropout_prob = 0.01
        # model.config.vocab_size = 100000
        # model.config.hidden_size = 1000

        deberta_config = model.config

        # deberta_config = DebertaV2Config(
        #     vocab_size=128000,  # 한국어 대규모 데이터셋을 위한 적절한 vocab size
        #     hidden_size=1024,  # 라지 모델의 히든 크기
        #     num_hidden_layers=24,  # 레이어 개수
        #     num_attention_heads=16,  # 어텐션 헤드 개수
        #     intermediate_size=4096,  # 피드포워드 레이어 크기
        #     max_position_embeddings=512,  # 최대 시퀀스 길이
        #     type_vocab_size=2,
        #     layer_norm_eps=1e-7,
        #     hidden_dropout_prob=0.1,
        #     attention_probs_dropout_prob=0.1,
        # )

        # deberta_config = DebertaV2Config(
        #     type_vocab_size=1,
        #     vocab_size=128100,
        #     hidden_size=1536,
        #     num_labels=2
        # )

        deberta_config.pad_token_id = 1
        # deberta_config.position_biased_input = False

        # model.config.max_position_embeddings = 768
        self.model = DebertaV2ForSequenceClassification(deberta_config)

        # self.multi_gpu = config.train.multi_gpu

        self.model.to(device)
        # summary(self.model, (4, 50))
        # self.model.apply(self.weights_init)

        self.criterion = nn.CrossEntropyLoss()
        self.softmax = nn.Softmax(dim=1)


        # self.optimizer = create_xadam(self.model, config.train.epoch)
        # self.optimizer = torch.optim.Adam(self.model.parameters(), lr=config.train.learning_rate)
        self.optimizer = torch.optim.AdamW(self.model.parameters(), lr=config.train.learning_rate)
        # torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)

        # self.pipe = pipeline(tokenizer=self.tokenizer, model=self.model, device=device)

        self.train_accuracy = []
        self.validation_accuracy = []

    def inference(self, inputs):
        self.model.eval()

        inputs = self.tokenizer(inputs, return_tensors="pt", padding=True).to(device)

        with torch.no_grad():
            logits = self.model(**inputs).logits

        predicted_class_id = logits.argmax(dim=1)

        return logits, predicted_class_id

    @staticmethod
    def weights_init(m):
        if isinstance(m, torch.nn.Linear):
            torch.nn.init.xavier_uniform_(m.weight)
            if m.bias is not None:
                torch.nn.init.zeros_(m.bias)

    def train_one(self, inputs, labels):
        self.optimizer.zero_grad()

        # print(inputs)
        inputs = self.tokenizer(inputs, return_tensors="pt", padding=True).to(device)

        # [ 배치사이즈가 1일때(kobert) ]
        # {'input_ids': tensor([[517,   0, 517,   0, 517, 493,   0, 490,   0, 517, 490,   0, 517,   0,
        #          517, 491, 494,   0, 517,   0, 517,   0, 517, 493,   0, 490,   0, 517,
        #            0, 517, 493,   0, 493,   0, 517,   0, 517, 493,   0, 517,   0, 490,
        #            0,   0,   0]]), 'token_type_ids': tensor([[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
        #          0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 2]]), 'attention_mask': tensor([[1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
        #          1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1]])}

        # [ 배치사이즈가 4일때(kobert) ]
        # ['개소리 공정 말 정권 들 바뀌', '판결 이번 되 이상', '반미 어떻 하 사건 세월호 이용 대통령 되 나라', '암 주연 끌 상관없']
        # {'input_ids': tensor([[  1,   1,   1,   1,   1,   1,   1,   1,   1,   1,   1,   1,   1, 517,
        #          490,   0, 494, 517, 490,   0, 517,   0, 517,   0, 490,   0, 517,   0,
        #          517,   0,   0,   0],
        #         [  1,   1,   1,   1,   1,   1,   1,   1,   1,   1,   1,   1,   1,   1,
        #            1,   1, 517,   0, 490,   0, 517, 491, 494,   0, 517,   0, 517, 491,
        #          494,   0,   0,   0],
        #         [517,   0, 494, 517, 491,   0, 517, 493,   0, 517,   0, 490,   0, 517,
        #            0, 491,   0, 493,   0, 517, 491, 494, 491,   0, 517,   0, 517,   0,
        #          517,   0,   0,   0],
        #         [  1,   1,   1,   1,   1,   1,   1,   1,   1,   1,   1,   1,   1,   1,
        #            1, 517, 491,   0, 517,   0, 491,   0, 517,   0, 517,   0, 490,   0,
        #          491,   0,   0,   0]]), 'token_type_ids': tensor([[3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
        #          0, 0, 0, 0, 0, 0, 0, 2],
        #         [3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 0, 0, 0, 0, 0, 0, 0, 0,
        #          0, 0, 0, 0, 0, 0, 0, 2],
        #         [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
        #          0, 0, 0, 0, 0, 0, 0, 2],
        #         [3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 0, 0, 0, 0, 0, 0, 0, 0, 0,
        #          0, 0, 0, 0, 0, 0, 0, 2]]), 'attention_mask': tensor([[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
        #          1, 1, 1, 1, 1, 1, 1, 1],
        #         [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1,
        #          1, 1, 1, 1, 1, 1, 1, 1],
        #         [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
        #          1, 1, 1, 1, 1, 1, 1, 1],
        #         [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1,
        #          1, 1, 1, 1, 1, 1, 1, 1]])}

        # 매핑 정보: kobert -> deberta
        # 0 -> 2 (시퀀스 끝, 3)
        # 0 ->
        # 배치 사이즈가 4일때 (microsoft deberta tokenizer)
        # {'input_ids': tensor([[     1,  96442, 106446,  97769, 122785,  68368,      2,      0,      0,
        #               0,      0,      0,      0,      0,      0,      0,      0,      0,
        #               0,      0],
        #         [     1,    507, 122863, 123001,  60388,  37060,  96152,  39027,  80163,
        #             507, 123272,  92388,    507,  67435, 114304,    507, 123947,  51223,
        #           15048,      2],
        #         [     1,    507, 123212,    507, 123212,    507, 122626,      2,      0,
        #               0,      0,      0,      0,      0,      0,      0,      0,      0,
        #               0,      0],
        #         [     1,    507, 123154,  42812,  51139,  64010,    507, 123460,  98088,
        #          122854,    507,      3, 113152,  73444,  98422,  52793, 105691,    507,
        #               3,      2]]), 'token_type_ids': tensor([[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        #         [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        #         [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        #         [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]]), 'attention_mask': tensor([[1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        #         [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
        #         [1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        #         [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1]])}

        # print(inputs)
        # print(inputs['input_ids'].shape)
        # if torch.isnan(inputs['input_ids']).any():
        #     raise Exception("input value has nan")

        # cudas = ["cuda:0", "cuda:1"]

        labels = labels.to(device)

        # output = self.model(input_ids=inputs['input_ids'], attention_mask=inputs['attention_mask'])
        output = self.model(**inputs)
        logits = output.logits

        # logits_with_softmax = self.softmax(logits)

        loss = self.criterion(logits, labels)

        # print(logits, logits_with_softmax, labels, loss.item())
        # if torch.isnan(loss).any():
        #     raise Exception("loss has nan")

        loss.backward()

        predicted_class_id = output.logits.argmax(dim=1)
        # print(predicted_class_id)

        correct = 0
        for predict, (zero, one) in zip(predicted_class_id, labels):
            if predict == one:
                correct += 1

        self.optimizer.step()

        return loss.item(), correct, len(labels)

    def vali_one(self, inputs, labels):
        inputs = self.tokenizer(inputs, return_tensors="pt", padding=True).to(device)

        labels = labels.to(device)

        with torch.no_grad():
            logits = self.model(**inputs).logits

        predicted_class_id = logits.argmax(dim=1)
        # print(logits, predicted_class_id, labels)
        # print(logits.shape, labels.shape)

        correct = 0
        for predict, (zero, one) in zip(predicted_class_id, labels):
            if predict == one:
                correct += 1

        return correct, len(labels)

    def train(self):
        self.model.train()

        losses = []
        corrects = 0
        total = 0
        losses_sum = 0
        for text, label in (pbar := tqdm(self.trainloader, ncols=100)):
            loss, correct, all = self.train_one(text, label)
            losses.append(loss)
            losses_sum += loss

            corrects += correct
            total += all
            pbar.set_description(f"{round(corrects / total * 100, 2)}%, loss: {round(losses_sum / len(losses), 2)}")

        train_accuracy = round(corrects / total * 100, 2)

        print("loss: {}, train accuracy: {}%".format(sum(losses) / len(losses), train_accuracy))

        return train_accuracy

    def validation(self):
        self.model.eval()

        counts = 0
        corrects = 0
        for text, label in tqdm(self.validationloader, ncols=100):
            correct, count = self.vali_one(text, label)

            corrects += correct
            counts += count

        validation_accuracy = round(corrects / counts * 100, 2)

        print("validation accuracy: {}%".format(validation_accuracy))

        return validation_accuracy

    def process(self, epoch=10, start_epoch=1):
        if start_epoch < 1:
            raise Exception("에포크는 1이상의 정수여야 합니다")

        if start_epoch > 1:
            self.load_weights(start_epoch - 1)  # 현재 시작할 에포크 - 1당시 값으로 설정 후 학습

        for i in range(start_epoch, epoch + 1):
            print("epoch {}/{}:".format(i, epoch))
            self.train_accuracy.append(self.train())
            self.validation_accuracy.append(self.validation())

            # torch.save(self.model, f'deberta_{i}.pt')
            self.save_weights(i)

        self.show_plot(self.train_accuracy, self.validation_accuracy)

    def load_weights(self, epoch):
        self.train_accuracy.clear()
        self.validation_accuracy.clear()

        checkpoint = torch.load(f'chkpt/deberta_{epoch}.pth', map_location=torch.device(device))

        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])

        if type(checkpoint['train_accuracy']) == float:
            self.train_accuracy.append(checkpoint['train_accuracy'])
            self.validation_accuracy.append(checkpoint['validation_accuracy'])
        else:
            self.train_accuracy.extend(checkpoint['train_accuracy'])
            self.validation_accuracy.extend(checkpoint['validation_accuracy'])

    def save_weights(self, epoch):
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'train_accuracy': self.train_accuracy,
            'validation_accuracy': self.validation_accuracy
        }

        os.makedirs('chkpt', exist_ok=True)
        torch.save(checkpoint, f'chkpt/deberta_{epoch}.pth')

    @staticmethod
    def show_plot(train_accuracies, validation_accuracies):
        print(train_accuracies)
        print(validation_accuracies)
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
    c = Config('config/config.yml')

    trainer = DebertaClassificationModel(c)
    trainer.process(
        epoch=c.train.epoch,
        start_epoch=c.train.start_epoch
    )

# pip3 freeze > requirements.txt
# pip install -r requirements.txt
