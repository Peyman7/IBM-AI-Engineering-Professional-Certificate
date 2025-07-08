# utils/data_utils.py
import torch
from torchtext.datasets import AG_NEWS
from torchtext.data.utils import get_tokenizer
from torchtext.vocab import build_vocab_from_iterator
from torch.utils.data import DataLoader, random_split
from torchtext.data.functional import to_map_style_dataset

tokenizer = get_tokenizer("basic_english")

def yield_tokens(data_iter):
    for _, text in data_iter:
        yield tokenizer(text.lower())

train_iter = AG_NEWS(split='train')
vocab = build_vocab_from_iterator(yield_tokens(train_iter), specials=["<unk>"])
vocab.set_default_index(vocab["<unk>"])

def text_pipeline(x): return vocab(tokenizer(x))
def label_pipeline(x): return int(x) - 1

def collate_batch(batch):
    label_list, text_list, offsets = [], [], [0]
    for _label, _text in batch:
        label_list.append(label_pipeline(_label))
        processed_text = torch.tensor(text_pipeline(_text), dtype=torch.int64)
        text_list.append(processed_text)
        offsets.append(processed_text.size(0))
    label_list = torch.tensor(label_list, dtype=torch.int64)
    offsets = torch.tensor(offsets[:-1]).cumsum(dim=0)
    text_list = torch.cat(text_list)
    return label_list, text_list, offsets

def get_dataloaders(batch_size, vocab, device):
    train_iter, test_iter = AG_NEWS()
    dataset = to_map_style_dataset(train_iter)
    num_train = int(len(dataset) * 0.95)
    train_data, valid_data = random_split(dataset, [num_train, len(dataset) - num_train])
    test_data = to_map_style_dataset(test_iter)

    return (
        DataLoader(train_data, batch_size=batch_size, shuffle=True, collate_fn=collate_batch),
        DataLoader(valid_data, batch_size=batch_size, shuffle=False, collate_fn=collate_batch),
        DataLoader(test_data, batch_size=batch_size, shuffle=False, collate_fn=collate_batch)
    )
