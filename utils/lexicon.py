from collections import defaultdict
import numpy as np
import torch
import torch.nn as nn
from utils.fasttext_load import get_direct_vector
def count_kmers_in_dataset_with_vocab(sequences, kmer_vocab, min_kmer_size=5, max_kmer_size=12):
    """
    统计数据集中所有k-mer的频率，使用已加载的k-mer词汇表。
    :param sequences: 输入批量序列 (list of str)
    :param kmer_vocab: 词汇表 (dict: {kmer: index})
    :param min_kmer_size: 最小k-mer大小 (int)
    :param max_kmer_size: 最大k-mer大小 (int)
    :return: k-mer频率字典 (dict: {kmer: frequency})
    """
    kmer_counts = defaultdict(int)
    
    for sequence in sequences:
        sequence_length = len(sequence)
        for kmer_size in range(min_kmer_size, max_kmer_size + 1):
            for i in range(sequence_length - kmer_size + 1):
                kmer = sequence[i:i + kmer_size]
                if kmer in kmer_vocab:
                    kmer_counts[kmer] += 1
    
    return kmer_counts



def readLexicon(file_path):
    kmer_vocab = {}
    with open(file_path, 'r') as f:
        # lines = f.readlines()[1:]  # 旧逻辑：无条件跳过表头；无表头 lexicon 会少读第一条 motif
        raw_lines = [line.strip() for line in f.readlines()]
        if raw_lines and raw_lines[0].lower() in {"kmer", "motif", "lexicon"}:
            lines = raw_lines[1:]
        else:
            lines = raw_lines
        for index, line in enumerate(lines):  # 使用enumerate自动生成索引
            # kmer = line.strip()  # 单列数据，只有kmer
            kmer = line  # 单列数据，只有kmer
            if not kmer:
                continue
            kmer_vocab[kmer] = index
    return kmer_vocab



#BME挂词
def match_and_embed_with_kmers_batch(sequences, kmer_vocab, total_kmer_counts,FastTextModel_path,kmer_embedding_cache, min_kmer_size=4,max_kmer_size=14):
    """
    对批量序列使用滑动窗口匹配多核苷酸片段（k-mer），并获取对应词嵌入。
    :param sequences: 输入批量序列 (list of str)
    :param kmer_vocab: 词汇表 (dict: {kmer: index})
    :param kmer_vectors: 词嵌入矩阵 (np.array)
    :param min_kmer_size: 最小k-mer大小 (int)
    :param max_kmer_size: 最大k-mer大小 (int)
    :return: 拼接后的嵌入 (torch.Tensor)
    """
    batch_size = len(sequences)
    max_seq_length = max(len(seq) for seq in sequences)
    vector_dim = 100
    
     # (4,180,3000)
     # 初始化嵌入矩阵，默认所有字符为零向量
    kmer_embeddings_batch = torch.zeros((batch_size, max_seq_length, vector_dim * 3))
    
    for b, sequence in enumerate(sequences):
        sequence_length = len(sequence)
        matched_kmer = [{'b': [], 'm': [], 'e': []} for _ in range(sequence_length)]
        kmer_freq = [{'b': [], 'm': [], 'e': []} for _ in range(sequence_length)]
        
        for kmer_size in range(min_kmer_size, max_kmer_size + 1):
            for i in range(sequence_length - kmer_size + 1):
                kmer = sequence[i:i + kmer_size]
                if kmer in kmer_vocab and kmer in total_kmer_counts:
                    idx = kmer_vocab[kmer]
                    # kmer_vector = torch.tensor(kmer_vectors[idx], dtype=torch.float)
                    # kmer_vector=get_direct_vector(kmer,FastTextModel_path)
                    kmer_vector = kmer_embedding_cache.get(kmer, None)
                    freq = total_kmer_counts[kmer]
                    for j in range(i, i + kmer_size):
                        if j == i:
                            matched_kmer[j]['b'].append(kmer_vector * freq)
                            kmer_freq[j]['b'].append(freq)
                        if j == i + kmer_size - 1:
                            matched_kmer[j]['e'].append(kmer_vector * freq)
                            kmer_freq[j]['e'].append(freq)
                        matched_kmer[j]['m'].append(kmer_vector * freq)
                        kmer_freq[j]['m'].append(freq)
        
        # 将多个kmer嵌入加权平均
        for i in range(sequence_length):
            total_freq = sum(kmer_freq[i]['b']) + sum(kmer_freq[i]['m']) + sum(kmer_freq[i]['e']) if (kmer_freq[i]['b'] or kmer_freq[i]['m'] or kmer_freq[i]['e']) else 1
            if matched_kmer[i]['b']:
                kmer_embeddings_batch[b, i, :vector_dim] = 4*(torch.sum(torch.stack(matched_kmer[i]['b']), dim=0) / total_freq )
            if matched_kmer[i]['m']:
                kmer_embeddings_batch[b, i, vector_dim:2 * vector_dim] = 4*(torch.sum(torch.stack(matched_kmer[i]['m']), dim=0) / total_freq )
            if matched_kmer[i]['e']:
                kmer_embeddings_batch[b, i, 2 * vector_dim:] = 4*(torch.sum(torch.stack(matched_kmer[i]['e']), dim=0) / total_freq )
    
    return kmer_embeddings_batch


def analyze_kmer_token_coverage(sequence, kmer_vocab, total_kmer_counts, min_kmer_size=4, max_kmer_size=14):
    """
    分析单个序列中哪些token被k-mer覆盖，并详细打印结果
    :param sequence: 输入序列 (str)
    :param kmer_vocab: 词汇表 (dict: {kmer: index})
    :param total_kmer_counts: k-mer频率字典
    :param min_kmer_size: 最小k-mer大小
    :param max_kmer_size: 最大k-mer大小
    """
    print(f"=== 分析序列: {sequence} ===")
    print(f"序列长度: {len(sequence)}")
    print()
    
    sequence_length = len(sequence)
    
    # 记录每个位置的token是否被k-mer覆盖
    token_coverage = [False] * sequence_length
    
    # 记录每个位置被哪些k-mer覆盖 (用于BME标记)
    token_kmer_info = [{'b': [], 'm': [], 'e': []} for _ in range(sequence_length)]
    
    # 记录所有匹配的k-mer
    matched_kmers = []
    
    # 遍历所有可能的k-mer
    for kmer_size in range(min_kmer_size, max_kmer_size + 1):
        for i in range(sequence_length - kmer_size + 1):
            kmer = sequence[i:i + kmer_size]
            if kmer in kmer_vocab and kmer in total_kmer_counts:
                freq = total_kmer_counts[kmer]
                matched_kmers.append({
                    'kmer': kmer,
                    'start': i,
                    'end': i + kmer_size - 1,
                    'length': kmer_size,
                    'frequency': freq,
                    'positions': list(range(i, i + kmer_size))
                })
                
                # 标记覆盖的token位置
                for j in range(i, i + kmer_size):
                    token_coverage[j] = True
                    # BME标记
                    if j == i:  # 开始位置
                        token_kmer_info[j]['b'].append(kmer)
                    if j == i + kmer_size - 1:  # 结束位置
                        token_kmer_info[j]['e'].append(kmer)
                    token_kmer_info[j]['m'].append(kmer)  # 中间位置（包括开始和结束）
    
    # 打印匹配到的k-mer信息
    print(f"找到 {len(matched_kmers)} 个匹配的k-mer:")
    print("-" * 80)
    for i, kmer_info in enumerate(matched_kmers, 1):
        print(f"{i:2d}. k-mer: {kmer_info['kmer']:15s} | "
              f"位置: {kmer_info['start']:3d}-{kmer_info['end']:3d} | "
              f"长度: {kmer_info['length']:2d} | "
              f"频率: {kmer_info['frequency']:6d}")
    
    print("\n" + "=" * 80)
    
    # 打印token覆盖情况
    print("Token覆盖情况:")
    print("位置:", " ".join(f"{i:2d}" for i in range(sequence_length)))
    print("Token:", " ".join(f" {sequence[i]}" for i in range(sequence_length)))
    print("覆盖:", " ".join(" ✓" if token_coverage[i] else " ✗" for i in range(sequence_length)))
    
    # 统计覆盖情况
    covered_count = sum(token_coverage)
    uncovered_count = sequence_length - covered_count
    coverage_rate = covered_count / sequence_length * 100
    
    print(f"\n覆盖统计:")
    print(f"- 被k-mer覆盖的token数量: {covered_count}/{sequence_length} ({coverage_rate:.1f}%)")
    print(f"- 未被k-mer覆盖的token数量: {uncovered_count}")
    
    # 打印未被覆盖的token位置
    uncovered_positions = [i for i in range(sequence_length) if not token_coverage[i]]
    if uncovered_positions:
        print(f"- 未被覆盖的token位置: {uncovered_positions}")
        print(f"- 未被覆盖的token: {[sequence[i] for i in uncovered_positions]}")
    else:
        print("- 所有token都被k-mer覆盖!")
    
    print("\n" + "=" * 80)
    
    # 详细打印每个位置的BME标记信息
    print("详细的BME标记信息:")
    print("(B=开始, M=中间, E=结束)")
    print("-" * 80)
    
    
    
    return {
        'matched_kmers': matched_kmers,
        'token_coverage': token_coverage,
        'token_kmer_info': token_kmer_info,
        'coverage_rate': coverage_rate,
        'covered_count': covered_count,
        'uncovered_positions': uncovered_positions
    }
