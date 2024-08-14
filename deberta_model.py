import argparse
import os

import torch
from datasets import tqdm
from kobert_tokenizer import KoBERTTokenizer
from matplotlib import pyplot as plt
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torchsummary import summary
from transformers import AutoTokenizer, DebertaV2ForSequenceClassification, DebertaV2Config, pipeline, \
    DebertaForSequenceClassification, RobertaForSequenceClassification
import torch.nn.functional as F
from yamlload import Config
from torch.nn.parallel import DistributedDataParallel as DDP

os.environ["TOKENIZERS_PARALLELISM"] = "false"


def collate_fn(batch):
    data = []
    labels = []

    # max_len = 0

    for text, label in batch:
        data.append(text)
        labels.append(label)
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
    def __init__(self, config, only_inference=False, distributed=False, gpu=None):
        batch_size = config.train.batch_size
        num_workers = config.train.num_workers

        if not only_inference:
            self.trainloader = DataLoader(dataset=TextLoader([config.data.train]),
                                          batch_size=batch_size,
                                          shuffle=config.train.shuffle,
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
        # model = DebertaV2ForSequenceClassification.from_pretrained("microsoft/deberta-v3-large")
        # model = RobertaForSequenceClassification.from_pretrained("FacebookAI/roberta-base")
        model = RobertaForSequenceClassification.from_pretrained("FacebookAI/roberta-large")
        # # model.config.max_position_embeddings = 1024
        # # del model.config.id2label[1]
        #
        # # self.model = DebertaForSequenceClassification(model.config).to(self.device)
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

        # global args

        # args.gpu = 0
        # args.world_size = 1

        # args.gpu = args.local_rank

        model_with_config = RobertaForSequenceClassification(deberta_config)

        if distributed:
            print(f'gpu: {gpu}')
            torch.cuda.set_device(gpu)
            torch.distributed.init_process_group(backend='nccl',
                                                 init_method='env://')
            # world_size = torch.distributed.get_world_size()

            model_with_config.cuda(gpu)
            self.model = DDP(model_with_config, delay_allreduce=True)
        else:
            self.model = model_with_config
            self.model.to(config.train.gpu)


        # self.multi_gpu = config.train.multi_gpu

        self.device = config.train.gpu if torch.cuda.is_available() else "cpu"
        self.model.to(self.device)
        # summary(self.model, (4, 50))
        # self.model.apply(self.weights_init)

        # self.criterion = nn.CrossEntropyLoss()
        self.criterion = BalancedFocalLoss(alpha=torch.tensor([0.25, 0.75]).to(self.device), gamma=2.0, weight=torch.tensor([1.0, 3.0]).to(self.device))
        self.softmax = nn.Softmax(dim=1)


        # self.optimizer = create_xadam(self.model, config.train.epoch)
        # self.optimizer = torch.optim.Adam(self.model.parameters(), lr=config.train.learning_rate)
        self.optimizer = torch.optim.AdamW(self.model.parameters(), lr=config.train.learning_rate)
        # torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)

        # self.pipe = pipeline(tokenizer=self.tokenizer, model=self.model, device=device)

        self.train_accuracy = []
        self.validation_accuracy = []

        del model

    def inference(self, inputs):
        self.model.eval()

        inputs = self.tokenizer(inputs, return_tensors="pt", padding=True).to(self.device)

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

    def count_correct_prediction(self, logits, labels):
        predicted_class_id = logits.argmax(dim=1)
        # print(logits.shape, labels.shape)
        correct = 0
        for predict, value in zip(predicted_class_id, labels):
            if predict == value:
                correct += 1

        return correct

    def train_one(self, inputs, labels):
        self.optimizer.zero_grad()

        inputs = self.tokenizer(inputs, return_tensors="pt", padding=True).to(self.device)
        labels = labels.to(self.device)

        output = self.model(**inputs).logits
        loss = self.criterion(output, labels)

        # if torch.isnan(loss).any():
        #     raise Exception("loss has nan")

        loss.backward()
        self.optimizer.step()

        return loss.item(), self.count_correct_prediction(output, labels), len(labels)

    def vali_one(self, inputs, labels):
        inputs = self.tokenizer(inputs, return_tensors="pt", padding=True).to(self.device)

        labels = labels.to(self.device)

        with torch.no_grad():
            output = self.model(**inputs).logits

        return self.count_correct_prediction(output, labels), len(labels)

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

        checkpoint = torch.load(f'chkpt/deberta_{epoch}.pth', map_location=torch.device(self.device))

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


class BalancedFocalLoss(nn.Module):
    def __init__(self, alpha=None, gamma=2.0, weight=None, reduction='mean'):
        """
        :param alpha: 양성 클래스의 가중치 (보통 0.25에서 0.75 사이).
        :param gamma: 초점 조정 매개변수 (보통 2.0).
        :param weight: 클래스 균형을 위한 가중치, 텐서 형태 [2] (이진 분류의 경우).
        :param reduction: 출력에 적용할 감소 방식: 'none' | 'mean' | 'sum'.
        """
        super(BalancedFocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.weight = weight
        self.reduction = reduction
        self.ce = nn.CrossEntropyLoss(weight=weight, reduction='none')

    def forward(self, inputs, targets):
        # BCE 손실 계산
        # bce_loss = F.binary_cross_entropy_with_logits(inputs, targets, weight=self.weight, reduction='none')
        ce_loss = self.ce(inputs, targets)

        # # 확률 예측 값 계산
        # probs = torch.sigmoid(inputs)
        #
        # # Focal Loss 구성 요소 계산
        # pt = probs * targets + (1 - probs) * (1 - targets)  # pt = p (y=1일 때), 아니면 1-p
        # focal_weight = (self.alpha * targets + (1 - self.alpha) * (1 - targets)) * ((1 - pt) ** self.gamma)

        # 예측 확률 계산 (Softmax를 통해)
        pt = torch.exp(-ce_loss)

        if self.alpha is not None:
            # ce_loss *= self.alpha.gather(0, targets[:, 1])
            ce_loss *= self.alpha.gather(0, targets)

        # Focal Loss 계산
        loss = (1 - pt) ** self.gamma * ce_loss

        # BCE와 Focal Loss 결합
        # loss = focal_weight * bce_loss
        # loss = bce_loss

        if self.reduction == 'mean':
            return loss.mean()
        elif self.reduction == 'sum':
            return loss.sum()
        else:
            return loss


# 학습 안시키면 정확도 51%
if __name__ == "__main__":
    c = Config('config/config.yml')

    parser = argparse.ArgumentParser()
    # FOR DISTRIBUTED:  Parse for the local_rank argument, which will be supplied
    # automatically by torch.distributed.launch.
    parser.add_argument("--local-rank", default=os.environ['LOCAL_RANK'], type=int)
    args = parser.parse_args()

    # args.local_rank = os.environ['LOCAL_RANK']
    args.distributed = False
    if 'WORLD_SIZE' in os.environ:
        args.distributed = int(os.environ['WORLD_SIZE']) > 1

    trainer = DebertaClassificationModel(
        c,
        distributed=args.distributed,
        gpu=args.local_rank,
    )
    trainer.process(
        epoch=c.train.epoch,
        start_epoch=c.train.start_epoch
    )

