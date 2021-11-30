from torch.utils.data import Dataset
import json

from SST_5.data_utils.vocab import get_vocab_by_strategy, token_wrapper


def load_file(filename):
    data = []
    with open(filename, "r") as f:
        for line in f.readlines():
            data.append(json.loads(line))
    return data


class SST5Dataset(Dataset):
    def __init__(self, dataset_type, data, tokenizer, args):
        super().__init__()
        self.args = args
        self.data = list()
        self.dataset_type = dataset_type
        self.x_hs, self.x_ts = [], []

        vocab = get_vocab_by_strategy(args, tokenizer)
        data_list=data.split('\n')
        for d in data_list:
            lable,text=d.split('\t')
            if token_wrapper(args, lable) not in vocab:
                continue
            self.x_ts.append(lable)
            self.x_hs.append(text)
            self.data.append(d)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, i):
        lable,text=self.data[i].split('\t')
        return text, lable
