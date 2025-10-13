from torch.utils.data import DataLoader
from torch.utils.data import dataset


class MyDataSet(dataset.Dataset):
    def __init__(self, data, label):
        self.data = data
        self.label = label

    def __getitem__(self, item):
        return self.data[item], self.label[item]

    def __len__(self):
        return len(self.data)
    
def SampleLoaderBERT(data, Label, BatchSize):
    Loader = DataLoader(
        dataset=MyDataSet(data, Label),
        batch_size=BatchSize,
        shuffle=True,
        num_workers=0,
        drop_last=False
    )
    return Loader