import scipy.io
import os
import Utils.OneHot
import pandas as pd

'''
    TFName represents name of TFs
    DataSetName represents name of Datasets
'''
def seq2kmer(seq):
    """
    Convert original sequence to kmers

    Arguments:
    seq -- str, original sequence.
    k -- int, kmer of length k specified.

    Returns:
    kmers -- str, kmers separated by space

    """
    kmer = [seq[x:x + 3] for x in range(len(seq) + 1 - 3)]
    kmers = " ".join(kmer)
    return kmers
import csv
def DataReader_prom(data_file_path, delimiter='\t'):
    FeatureMatrix = []
    DenseLabels = []
    with open(data_file_path, 'r') as file:
            reader = csv.reader(file, delimiter=delimiter)
            for row in reader:
                # 跳过空行
                if not row:
                    continue
                
                # 确保每行有 2 列（序列和标签）
                if len(row) != 2:
                    raise ValueError(f"无效的行格式，预期 2 列，实际 {len(row)} 列: {row}")
                
                sequence, labels = row
                sequence = sequence.strip()
                labels = labels.strip()
                
                # 验证序列
                if not sequence:
                    raise ValueError("发现空序列")
                if not all(c in 'ACGT' for c in sequence):
                    raise ValueError(f"序列中包含无效核苷酸: {sequence}")
                
                # 验证标签
                if not labels:
                    raise ValueError("发现空标签")
                try:
                    label_list = list(map(int, labels))
                except ValueError:
                    raise ValueError(f"无效的标签格式: {labels}")
                
                # 检查序列和标签长度是否一致
                if len(sequence) != len(label_list):
                    raise ValueError(f"序列长度 ({len(sequence)}) 与标签长度 ({len(label_list)}) 不匹配")
                
                # 添加到列表
                FeatureMatrix.append(sequence)
                DenseLabels.append(label_list)
            
            # 检查是否读取到数据
            if not FeatureMatrix:
                raise ValueError("文件中没有有效数据")
            
            # 检查标签长度一致性
            label_length = len(DenseLabels[0])
            if not all(len(labels) == label_length for labels in DenseLabels):
                raise ValueError("标签长度不一致")
            
            # 将序列转换为 one-hot 编码
            FeatureMatrix = Utils.OneHot.OneHot(
                sequence=FeatureMatrix,
                number=len(FeatureMatrix),
                nucleotide=4,
                length=label_length
            )
            
            return FeatureMatrix, DenseLabels
  
   

def DataReader(TFName, DataSetName, type):
    FeatureMatrix = []
    DenseLabels = []
    if DataSetName == 'ChIP-exo':
        '''
            AR -> labels
            GR -> label
        '''
        DenseLabels = scipy.io.loadmat(os.path.dirname(__file__) + '/ChIP-exo/' + TFName + '/label.mat')['labels']
        with open(os.path.dirname(__file__) + '/ChIP-exo/' + TFName + '/sequence.txt', 'r') as SReader:
            for line in SReader.readlines():
                FeatureMatrix.append(list(map(str, line.rstrip('\n'))))
        FeatureMatrix = Utils.OneHot.OneHot(sequence=FeatureMatrix, number=len(FeatureMatrix), nucleotide=4,
                                            length=DenseLabels.shape[1])
    else:
        with open(os.path.dirname(__file__) + '/ChIP-seq/' + TFName + '/baseline/seq_' + type + '.txt', 'r') as SReader, open(
                  os.path.dirname(__file__) + '/ChIP-seq/' + TFName + '/baseline/lab_' + type + '.txt', 'r') as LReader:
            for SRLine, LRLine in zip(SReader.readlines(), LReader.readlines()):
                FeatureMatrix.append(SRLine.rstrip('\n'))
                DenseLabels.append(list(map(int, [label for label in LRLine.rstrip('\n')])))
        FeatureMatrix = Utils.OneHot.OneHot(sequence=FeatureMatrix, number=len(FeatureMatrix), nucleotide=4,
                                            length=len(DenseLabels[1]))
    return FeatureMatrix, DenseLabels

def DataReaderPrecit(path):

    Dataset = pd.read_csv(path, header=None)
    Sequence = Dataset[0].tolist()
    FeatureMatrix = Utils.OneHot.OneHot(sequence=Sequence, number=len(Sequence), nucleotide=4,
                                        length=100)
    return FeatureMatrix


def DataReaderBERT(TFName, DataSetName, KMER):

    trainDataset = pd.read_csv('../Dataset/' + DataSetName + '/' + TFName + '/' + str(KMER) + '-mer/train.txt', sep='\t')

    TrainSequence = trainDataset['sequence'].tolist()
    TrainDenseLabels = []
    TrainLabels = trainDataset['label'].tolist()

    for row in trainDataset['denseLabel']:
        TrainDenseLabels.append(list(map(int, [label for label in row])))

    testDataset = pd.read_csv('../Dataset/' + DataSetName + '/' + TFName + '/' + str(KMER) + '-mer/test.txt', sep='\t')

    TestSequence = testDataset['sequence'].tolist()
    TestDenseLabels = []
    TestLabels = testDataset['label'].tolist()

    for row in testDataset['denseLabel']:
        TestDenseLabels.append(list(map(int, [label for label in row])))

    return TrainSequence, TrainDenseLabels, TrainLabels, TestSequence, TestDenseLabels, TestLabels

def DataReaderSequence(TFName, DataSetName, KMER):

    trainDataset = pd.read_csv('../Dataset/' + DataSetName + '/' + TFName + '/' + str(KMER) + '-mer/train.txt', sep='\t')

    TrainSequence = trainDataset['sequence'].tolist()
    TrainLabels = trainDataset['label'].tolist()

    testDataset = pd.read_csv('../Dataset/' + DataSetName + '/' + TFName + '/' + str(KMER) + '-mer/test.txt', sep='\t')

    TestSequence = testDataset['sequence'].tolist()
    TestLabels = testDataset['label'].tolist()

    return TrainSequence, TrainLabels, TestSequence, TestLabels

def DataReaderPrecitBERT(path):

    Dataset = pd.read_csv(path, header=None)
    Sequence = Dataset[0].tolist()
    Sequence = list(map(seq2kmer, Sequence))

    return Sequence

