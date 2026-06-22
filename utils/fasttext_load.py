import fasttext
from Bio import SeqIO
import torch
import numpy as np
# fasttext_model=None
fasttext_models={}


def load_model(FastTextModel_path):
    global fasttext_models
    if FastTextModel_path not in fasttext_models:
        fasttext_models[FastTextModel_path] = fasttext.load_model(FastTextModel_path)
    return fasttext_models[FastTextModel_path]

def generate_ngrams(sequence, min_n=1, max_n=10):
    """生成 1 到 10 的 N-gram"""
    ngrams = []
    seq = str(sequence).upper()
    for n in range(min_n, max_n + 1):
        for i in range(len(seq) - n + 1):
            ngrams.append(seq[i:i + n])
    return ngrams



def get_direct_vector(sequence, FastTextModel_path, normalize=False):
    """
    为序列生成 FastText 特征，不使用 N-gram，不填充，直接输出原生维度
    自动适应单个序列或批量序列输入
    Args:
        sequence (str or list): 单个 DNA 序列（如 "TAGATGCTCC"）或序列列表
        fasttext_model: 加载的 FastText 模型实例
        normalize (bool): 是否对向量进行 L2 归一化，默认为 True
    Returns:
        torch.Tensor: 单个序列返回形状 (native_dim,) 的向量，批量返回 (batch_size, native_dim)
    """
    model=load_model(FastTextModel_path)
    # 统一处理输入为列表形式
    if isinstance(sequence, str):
        sequences = [sequence]
        single_input = True
    else:
        sequences = sequence
        single_input = False
    
    batch_size = len(sequences)
    native_dim = model.get_dimension()  # FastText 原生维度（通常 100）
    
    # 初始化特征张量
    direct_vectors = np.zeros((batch_size, native_dim), dtype=np.float32)
    
    # 处理每个序列
    for batch_idx, seq in enumerate(sequences):
        seq = str(seq).upper()
        if not seq:  # 跳过空序列
            continue
            
        # 直接获取 FastText 原生嵌入
        vector = model.get_word_vector(seq)  # 输出 native_dim 维
        direct_vectors[batch_idx] = vector
    
    # 可选：L2 归一化
    if normalize:
        norms = np.linalg.norm(direct_vectors, axis=1, keepdims=True)
        direct_vectors = direct_vectors / (norms + 1e-10)
    
    # 转换为 PyTorch 张量
    result = torch.tensor(direct_vectors, dtype=torch.float32)
    
    # 对于单个序列，移除 batch 维度
    return result[0] if single_input else result

# # 使用示例
# if __name__ == "__main__":
#     # 单个序列
#     single_seq = "ATCG"
#     single_result = get_direct_vector(single_seq, model)
#     print("Single sequence shape:", single_result.shape)  # 输出 (1000,)
#     print("Single vector norm:", torch.norm(single_result).item())  # 检查归一化
   
#     # 批量序列
#     batch_seqs = ["ATCG", "GCTA", "TTT"]
#     batch_result = get_direct_vector(batch_seqs, model)
#     print("Batch sequence shape:", batch_result.shape)  # 输出 (3, 1000)