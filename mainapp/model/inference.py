import torch
from transformers import AutoTokenizer, DebertaForSequenceClassification

device = "cuda" if torch.cuda.is_available() else "cpu"


class DebertaInference:
    def __init__(self):
        self.checkpoint = "deberta.pt"

        # model.config
        self.tokenizer = AutoTokenizer.from_pretrained("microsoft/deberta-base")
        # model = DebertaForSequenceClassification.from_pretrained("microsoft/deberta-base").to(device)
        # model.config.max_position_embeddings = 1024
        # del model.config.id2label[1]

        # self.model = DebertaForSequenceClassification(model.config).to(device)
        # num_labels = len(model.config.id2label)
        self.model = DebertaForSequenceClassification.from_pretrained("microsoft/deberta-base",
                                                                      num_labels=2).to(device)

    def inference(self, text):
        inputs = self.tokenizer(text, return_tensors="pt", padding=True).to(device)

        with torch.no_grad():
            logits = self.model(**inputs).logits

        return logits[0], logits[1], logits.argmax().item()
