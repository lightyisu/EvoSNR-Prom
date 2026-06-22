import pandas as pd
# trainDataset = "/data/zwk/Evosnr-main/processed_datasets/train_set.csv"
# devDataset="/data/zwk/Evosnr-main/processed_datasets/validation_set.csv"
label2id = {'0': 0, '1': 1}
def readfileCSV(input_file):
    sequences = []  # To store sequences
    annotations = []  # To store annotations
    # 读取验证集（开发集）
    df = pd.read_csv(input_file, encoding='utf-8')
    # 验证列是否存在
    required_columns = ['Sequence', 'Annotation']
    if all(col in df.columns for col in required_columns):
        # 遍历处理每一行数据
        for index, row in df.iterrows():
            sequence = row['Sequence']
            annotation = row['Annotation']
            ann_numeric=[label2id[label] for label in annotation]
            sequences.append(sequence)
            annotations.append(ann_numeric)

    
    return sequences, annotations
import pandas as pd

label2id = {'0': 0, '1': 1}

def readfile(input_file):
    sequences = []
    annotations = []
    
    # 使用制表符分隔，无标题行，并指定列名
    df = pd.read_csv(input_file, 
                    sep='\t',          # 指定分隔符为制表符
                    header=None,       # 文件没有标题行
                    names=['Sequence', 'Annotation'])  # 手动指定列名
    
    # 验证列是否正确
    required_columns = ['Sequence', 'Annotation']
    if all(col in df.columns for col in required_columns):
        for index, row in df.iterrows():
            sequence = row['Sequence']
            annotation = row['Annotation']
             # 跳过无效的 sequence：空值、None、NaN 等
            if pd.isna(sequence) or not sequence:  # 判断是否为 NaN 或空字符串
                    continue
            # 将标签字符串转换为数字列表
            ann_numeric = [label2id[label] for label in annotation]
            
            sequences.append(sequence)
            annotations.append(ann_numeric)
    else:
        print("文件格式错误，缺少必要的列！")
        return [], []
    
    return sequences, annotations
def readfileLev(input_file):
    """
    从指定路径的文件中读取序列和注释。
    文件格式应为每行 '序列,注释'，例如:
    GTCTATCCTCAAAAATAAATCAGGCTGGTTTGTCAGGTCTAGGTGTCGCTAACACGGGCGCCTAGTTGATAGTGATATACT,0
    """
    sequences = []
    annotations = []

    # 将输入的文件路径赋值给 file_path
    file_path = input_file

    try:
        # 使用 with 语句确保文件在使用后自动关闭
        with open(file_path, 'r', encoding='utf-8') as infile:
            # 按行读取文件
            for line in infile:
                # strip() 方法去除行首尾的空白字符（包括换行符）
                line = line.strip()
                if line: # 确保不是空行
                    # 使用 split(',') 按逗号分割字符串
                    # 这会将行 'sequence,annotation' 分割成 ['sequence', 'annotation']
                    parts = line.split(',')

                    # 检查分割后是否正好是两部分（序列和注释）
                    if len(parts) == 2:
                        sequence = parts[0]
                        try:
                            # 将注释部分转换为整数
                            annotation = int(parts[1])
                            sequences.append(sequence)
                            annotations.append(annotation)
                        except ValueError:
                            # 如果注释部分不能转换为整数，打印警告并跳过该行
                            print(f"Warning: Annotation part '{parts[1]}' is not an integer. Skipping line: {line}")
                    else:
                        # 处理格式不正确的行
                        # 例如，如果行中没有逗号或有多于一个逗号
                        print(f"Warning: Skipping line with incorrect format (expected 'sequence,annotation'): {line}")

    except FileNotFoundError:
        print(f"Error: The file '{file_path}' was not found.")
    except Exception as e:
        print(f"An error occurred while reading the file: {e}")

    return sequences, annotations

#two seq
def DataReaderBERT(path1,path2):
    # Read training and validation datasets
    TrainSequence, TrainDenseLabels = readfile(path1)
    TestSequence, TestDenseLabels = readfile(path2)
    
    return TrainSequence, TrainDenseLabels, TestSequence, TestDenseLabels
#one seq
def DataReaderBERT_SingleDomain(path):
    # Read training and validation datasets
    TrainSequence, TrainDenseLabels = readfile(path)
    
    
    return TrainSequence, TrainDenseLabels

def DataReaderBERT_SeqLev(domain_path):
    # Read training and validation datasets
    TrainSequence, TrainDenseLabels = readfileLev(domain_path)
    
    
    return TrainSequence, TrainDenseLabels

# ts,td,tes,ted=DataReaderBERT()
# print(td[:2])
