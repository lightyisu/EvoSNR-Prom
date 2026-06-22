from Dataset import MyDataSet
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
    
def SampleLoader(FeatureMatrix, DenseLabel, BatchSize):
    Loader = DataLoader(
        dataset=MyDataSet(FeatureMatrix, DenseLabel),
        batch_size=BatchSize,
        shuffle=True,
        num_workers=0,
        drop_last=False
    )
    return Loader

def SampleLoaderPredict(FeatureMatrix, BatchSize):
    Loader = DataLoader(
        dataset=MyDataSet.MyDataSetPredict(FeatureMatrix),
        batch_size=BatchSize,
        shuffle=True,
        num_workers=0,
        drop_last=False
    )
    return Loader

# def SampleLoaderBERT(Sequence, DenseLabel, Label, BatchSize):
#     Loader = DataLoader(
#         dataset=MyDataSet.MyDataSetBERT(Sequence, DenseLabel, Label),
#         batch_size=BatchSize,
#         shuffle=True,
#         num_workers=0,
#         drop_last=False
#     )
#     return Loader
def SampleLoaderBERT(data, Label, BatchSize):
    Loader = DataLoader(
        dataset=MyDataSet(data, Label),
        batch_size=BatchSize,
        shuffle=True,
        num_workers=0,
        drop_last=False
    )
    return Loader

def SampleLoaderSequence(Sequence, Label, BatchSize):
    Loader = DataLoader(
        dataset=MyDataSet.MyDataSetSequence(Sequence, Label),
        batch_size=BatchSize,
        shuffle=True,
        num_workers=0,
        drop_last=False
    )
    return Loader

def SampleLoaderPredictBERT(Sequence, DenseLabel, Label, BatchSize):
    Loader = DataLoader(
        dataset=MyDataSet.MyDataSetBERT(Sequence, DenseLabel, Label),
        batch_size=BatchSize,
        shuffle=False,
        num_workers=0,
        drop_last=False
    )
    return Loader

def SampleLoaderPredictUnlabelBERT(Sequence, BatchSize):
    Loader = DataLoader(
        dataset=MyDataSet.MyDataSetPredictBERT(Sequence),
        batch_size=BatchSize,
        shuffle=False,
        num_workers=0,
        drop_last=False
    )
    return Loader