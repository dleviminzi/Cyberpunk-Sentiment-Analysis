from transformers import BertForSequenceClassification
import torch.nn as nn

class BERT(nn.Module):
    # implementation is made extremely easy by huggingface
    def __init__(self):
        super(BERT, self).__init__()

        options_name = "bert-base-uncased"
        self.encoder = BertForSequenceClassification.from_pretrained(options_name)

    def forward(self, text, label, training=True):
        if training:
            loss, prediction = self.encoder(text, labels=label)[:2]
            return loss, prediction
        else:
            prediction = self.encoder(text, labels=label)[:2]
            return prediction