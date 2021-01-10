from torchtext import data
from transformers import BertTokenizerFast
import torch


def load_datasets():
    # initialize bert fast (rust) tokenizer from hugging face
    tokenizer = BertTokenizerFast.from_pretrained('bert-base-uncased')

    # pad: [PAD] and unk: [UNK] by default, so convert to indices 
    PAD = tokenizer.convert_tokens_to_ids(tokenizer.pad_token)
    UNK = tokenizer.convert_tokens_to_ids(tokenizer.unk_token)

    # using torchtext to load in dataset
    sentiment = data.Field(sequential=False, 
                       use_vocab=False, 
                       batch_first=True, 
                       dtype=torch.float)

    review = data.Field(use_vocab=False, 
                        tokenize=tokenizer.encode,
                        lower=False, 
                        include_lengths=False, 
                        batch_first=True, 
                        pad_token=PAD, 
                        unk_token=UNK)

    fields = [
        ('t_review', review),
        ('sentiment', sentiment)
    ]

    # load training/validation datasets
    train_ds, validate_ds = data.TabularDataset.splits(path='./data', 
                                                       train='train.csv', 
                                                       validation='validate.csv', 
                                                       format='csv', 
                                                       fields=fields, 
                                                       skip_header=True)
    
    print("Finished loading datasets.")     # let me know we're still kicking

    return train_ds, validate_ds


def get_iterators():
    train_ds, validate_ds = load_datasets()

    # work on cpu or gpu (this will run SLOW on cpu)
    device = 'cpu'
    if torch.cuda.is_available():
        device = 'cuda:0'

    # training, validation, and testing iterators
    tr_iter, val_iter = data.BucketIterator.splits(datasets=(train_ds, validate_ds),
                                                   sort_key=lambda x: len(x.t_review), 
                                                   device=device, 
                                                   batch_sizes=(16, 16),
                                                   sort_within_batch=True, 
                                                   repeat=False)

    #tst_iter = data.Iterator(test_ds, batch_size=64, device=device, train=False,
     #                        shuffle=False, sort=False)

    return tr_iter, val_iter
