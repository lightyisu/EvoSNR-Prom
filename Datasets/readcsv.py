import pandas as pd

# 读取数据集
trainDataset = "/data/zwk/yh_datasets/processed_datasets/train_set.csv"
devDataset = "/data/zwk/yh_datasets/processed_datasets/validation_set.csv"

# 读取验证集（开发集）
df = pd.read_csv(devDataset, encoding='utf-8')

# 验证列是否存在
required_columns = ['Sequence', 'Annotation']
if all(col in df.columns for col in required_columns):
    # 遍历处理每一行数据
    for index, row in df.iterrows():
        sequence = row['Sequence']
        annotation = row['Annotation']
        
        # 示例处理逻辑（可根据实际需求修改）
        print(f"样本 {index}:")
        print(f"Sequence 长度: {len(sequence)}")
        print(f"Annotation 类型: {type(annotation)}")
        
        # 这里可以添加具体的处理逻辑，例如：
        # 1. 序列预处理（去除空格/特殊字符）
        # 2. 注释解析（拆分成多个标签）
        # 3. 数据转换（序列编码为数值向量）
        
    # 可选：创建新列存储处理结果
    df['Processed_Seq'] = df['Sequence'].apply(lambda x: x.strip().upper())
    df['Annotation_List'] = df['Annotation'].str.split(';')
    
    # 展示处理后的 DataFrame
    print("\n处理后的数据集预览:")
    print(df[['Sequence', 'Annotation', 'Processed_Seq', 'Annotation_List']].head())
    
else:
    missing_cols = [col for col in required_columns if col not in df.columns]
    print(f"错误：缺失必要列 {missing_cols}，请检查数据集结构")