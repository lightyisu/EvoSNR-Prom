

from tqdm import tqdm
import os
os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import transformers.optimization as op
import pandas as pd
import gc

from collections import defaultdict
from torch.utils.data import Dataset, DataLoader, Sampler
from torch.cuda.amp import autocast

import Datasets.DataLoader
import Datasets.DataReader_EVO

# 新模型（融合了词汇融合 + 迁移学习）
import models.model_evosnr


from utils.compute_metrics import compute_metrics
from utils.seed import set_seed
from utils.save_load import save_model_evosnr, load_model_evosnr
from utils.early_stop import EarlyStopping
from utils.count_lexicon import countLexiconKmer
from utils.lexicon import (
    count_kmers_in_dataset_with_vocab,
    readLexicon,
    match_and_embed_with_kmers_batch,
)
from utils.fasttext_load import get_direct_vector
from utils.bash_config import parse_args


# ===========================================================
# 配置加载（与各 train_*.py 保持一致）
# ===========================================================
exec(open("configurator.py").read())

args = parse_args()

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# ---------- 超参数 ----------
MaxEpoch = 60
BatchSize = 2         # 必须为偶数（DualRepeatBalancedBatchSampler 要求）
MaxStepsPerEpoch = 50000
ThresholdValue = 0.5

# ---------- 词汇表 & FastText ----------
# kmer_vocab = readLexicon(Lexicon_target_path)  # 旧逻辑：只使用目标域词表
source_kmer_vocab = readLexicon(Lexicon_source_path)
target_kmer_vocab = readLexicon(Lexicon_target_path)
kmer_vocab = {
    kmer: index
    for index, kmer in enumerate(sorted(set(source_kmer_vocab) | set(target_kmer_vocab)))
}
print(
    f"词汇表大小: {len(kmer_vocab)} "
    f"(source={len(source_kmer_vocab)}, target={len(target_kmer_vocab)}, union={len(kmer_vocab)})"
)

# ---------- 数据路径 ----------
source_path = Source_PCA_path
dev_path    = PCA_split_path + "split_dev.csv"
train_path  = PCA_split_path + "split_train.csv"
test_path   = PCA_split_path + "split_test.csv"

# ---------- 损失函数 ----------



# ===========================================================
# Dataset 类（来自 train_pure_tl.py）
# ===========================================================
class NERDataset(Dataset):
    def __init__(self, sentences, tags):
        self.sentences = sentences
        self.tags = tags

    def __len__(self):
        return len(self.sentences)

    def __getitem__(self, idx):
        return self.sentences[idx], self.tags[idx]


# ===========================================================
# 双域均衡采样器（完全复刻自 train_pure_tl.py）
# ===========================================================
class DualRepeatBalancedBatchSampler(Sampler):
    """
    每个 batch 中，一半来自源域，一半来自目标域。
    当目标域数据量不足时，循环重复采样。
    """
    def __init__(self, source_dataset, target_dataset, batch_size):
        self.source_dataset = source_dataset
        self.target_dataset = target_dataset
        self.batch_size = batch_size

        if batch_size % 2 != 0:
            raise ValueError("BatchSize must be even to ensure balanced source/target batches")

        self.source_len = len(source_dataset)
        self.target_len = len(target_dataset)
        self.num_batches = int(np.ceil(self.source_len / (batch_size // 2)))
        self.target_repeats = max(
            1,
            int(np.ceil((self.num_batches * (batch_size // 2)) / self.target_len)),
        )

    def __iter__(self):
        source_indices = torch.randperm(self.source_len).tolist()

        target_indices = []
        for _ in range(self.target_repeats):
            target_indices.extend(torch.randperm(self.target_len).tolist())
        target_indices = target_indices[: self.num_batches * (self.batch_size // 2)]

        for i in range(self.num_batches):
            s_start = i * (self.batch_size // 2)
            s_end   = s_start + (self.batch_size // 2)
            s_batch = source_indices[s_start:s_end]
            if len(s_batch) < self.batch_size // 2:
                needed  = self.batch_size // 2 - len(s_batch)
                s_batch += source_indices[:needed]

            t_start = i * (self.batch_size // 2)
            t_end   = t_start + (self.batch_size // 2)
            t_batch = target_indices[t_start:t_end]

            yield s_batch, t_batch

    def __len__(self):
        return self.num_batches


# ===========================================================
# 数据加载
# ===========================================================
print(f"Loading source: {source_path}")
print(f"Loading target train: {train_path}")

(
    SourceSequenceRaw, SourceLabelRaw,
    TargetTrainSequenceRaw, TargetTrainLabelRaw,
) = Datasets.DataReader_EVO.DataReaderBERT(path1=source_path, path2=train_path)

SourceSequence      = np.array(SourceSequenceRaw)
SourceLabel         = np.array(SourceLabelRaw)
TargetTrainSequence = np.array(TargetTrainSequenceRaw)
TargetTrainLabel    = np.array(TargetTrainLabelRaw)

(
    TargetDevSequenceRaw, TargetDevLabelRaw,
    TargetTestSequenceRaw, TargetTestLabelRaw,
) = Datasets.DataReader_EVO.DataReaderBERT(path1=dev_path, path2=test_path)

TargetDevSequence  = np.array(TargetDevSequenceRaw)
TargetDevLabel     = np.array(TargetDevLabelRaw)
TargetTestSequence = np.array(TargetTestSequenceRaw)
TargetTestLabel    = np.array(TargetTestLabelRaw)

# ---------- 统计 k-mer 频率（来自 train_lexicon.py）----------
# ---------- 统计目标域 k-mer 频率（来自 train_lexicon.py）----------  # 旧说明：只统计目标域
# 使用目标域训练集 + 测试集统计词汇出现频次，用于 FastText 嵌入过滤  # 旧说明
# 使用源域训练集 + 目标域训练/测试集统计 union 词汇出现频次，用于 FastText 嵌入过滤
total_kmer_counts = defaultdict(int)

# 构建临时 DataLoader 用于频率统计
_SourceDenseLabels = torch.tensor(SourceLabelRaw)
_SourceLoader_stat = Datasets.DataLoader.SampleLoaderBERT(
    data=SourceSequence, Label=_SourceDenseLabels, BatchSize=BatchSize
)
_TrainDenseLabels = torch.tensor(TargetTrainLabelRaw)
_TrainLoader_stat = Datasets.DataLoader.SampleLoaderBERT(
    data=TargetTrainSequence, Label=_TrainDenseLabels, BatchSize=BatchSize
)
_TestDenseLabels = torch.tensor(TargetTestLabelRaw)
_TestLoader_stat = Datasets.DataLoader.SampleLoaderBERT(
    data=TargetTestSequence, Label=_TestDenseLabels, BatchSize=BatchSize
)
total_kmer_counts = countLexiconKmer(tqdm(_SourceLoader_stat), kmer_vocab, total_kmer_counts)
total_kmer_counts = countLexiconKmer(tqdm(_TrainLoader_stat), kmer_vocab, total_kmer_counts)
total_kmer_counts = countLexiconKmer(tqdm(_TestLoader_stat),  kmer_vocab, total_kmer_counts)

# ---------- 预构建 FastText 词汇缓存（来自 train_lexicon.py）----------
target_kmer_embedding_cache = {
    kmer: get_direct_vector(kmer, FastTextModel_target_path)
    for kmer in kmer_vocab
    if kmer in total_kmer_counts
}

# 源域使用同一词汇表 + 源域 FastText 模型
source_kmer_embedding_cache = {
    kmer: get_direct_vector(kmer, FastTextModel_source_path)
    for kmer in kmer_vocab
    if kmer in total_kmer_counts
}

# ---------- 目标域验证/测试 DataLoader（与 train_lexicon.py 完全一致）----------
DevDenseLabels  = torch.tensor(TargetDevLabelRaw)
TestDenseLabels = torch.tensor(TargetTestLabelRaw)

DevLoader  = Datasets.DataLoader.SampleLoaderBERT(
    data=TargetDevSequence, Label=DevDenseLabels, BatchSize=BatchSize
)
TestLoader = Datasets.DataLoader.SampleLoaderBERT(
    data=TargetTestSequence, Label=TestDenseLabels, BatchSize=BatchSize
)

mode = "train"


# ===========================================================
# main
# ===========================================================
def main():
    try:
        if mode == "train":
            print("Enter the Train Mode --------------------->")
            seed = getattr(args, "seed", 1)
            set_seed(seed)

            # -------- 模型初始化 --------
            neural_network = models.model_evosnr.EvoSNR().to(device)

            # -------- 优化器（三组学习率，来自 train_lexicon.py）--------
            evo_params     = list(map(id, neural_network.evo.parameters()))
            fusion_params  = list(map(id, neural_network.gated_fusion.parameters()))
            downstream_params = filter(
                lambda p: id(p) not in evo_params and id(p) not in fusion_params,
                neural_network.parameters(),
            )

            optimizer_grouped_parameters = [
                {"params": neural_network.evo.parameters(),          "lr": 5e-5},   # EVO: 小学习率
                {"params": neural_network.gated_fusion.parameters(), "lr": 1e-4},   # 门控融合: 中等
                {"params": downstream_params,                        "lr": 2e-4},   # 下游: 大
            ]

            # -------- Dataset & Sampler（来自 train_pure_tl.py）--------
            source_dataset    = NERDataset(SourceSequenceRaw, SourceLabelRaw)
            target_dataset    = NERDataset(TargetTrainSequenceRaw, TargetTrainLabelRaw)
            target_val_dataset = NERDataset(TargetDevSequenceRaw, TargetDevLabelRaw)

            sampler = DualRepeatBalancedBatchSampler(source_dataset, target_dataset, BatchSize)
            # t_total = len(SourceSequence) * MaxEpoch // BatchSize  # 旧逻辑：未考虑 MaxStepsPerEpoch 截断，步数会偏大
            steps_per_epoch = min(len(sampler), MaxStepsPerEpoch)
            t_total = steps_per_epoch * MaxEpoch

            optimizer = optim.AdamW(optimizer_grouped_parameters, eps=1e-8, betas=(0.9, 0.999))
            scheduler = op.get_linear_schedule_with_warmup(
                optimizer,
                num_warmup_steps=int(t_total * 0.1),
                num_training_steps=t_total,
            )

            print(f"Seed: {seed} 🌙")
            early_stopping = EarlyStopping(patience=25)
            best_mcc = -float("inf")

            # 验证集用 SampleLoaderBERT（与 train_lexicon.py 完全一致）
            ValidProgressBar = tqdm(DevLoader)

            # ====== 训练循环 ======
            for epoch in range(MaxEpoch):
                neural_network.train()
                total_loss  = 0.0
                num_batches = min(len(sampler), MaxStepsPerEpoch)

                with tqdm(total=num_batches, desc=f"Epoch {epoch + 1}/{MaxEpoch}", unit="batch") as pbar:
                    for i, (s_indices, t_indices) in enumerate(sampler):
                        if i >= num_batches:
                            break

                        source_batch = [source_dataset[idx] for idx in s_indices]
                        target_batch = [target_dataset[idx] for idx in t_indices]

                        # 提取序列，用于计算 k-mer 嵌入
                        source_sentences = [item[0] for item in source_batch]
                        target_sentences = [item[0] for item in target_batch]

                        # 计算 k-mer 词汇嵌入（来自 train_lexicon.py）
                        source_kmer_emb = match_and_embed_with_kmers_batch(
                            source_sentences,
                            kmer_vocab,
                            total_kmer_counts,
                            FastTextModel_source_path,
                            source_kmer_embedding_cache,
                        )
                        target_kmer_emb = match_and_embed_with_kmers_batch(
                            target_sentences,
                            kmer_vocab,
                            total_kmer_counts,
                            FastTextModel_target_path,
                            target_kmer_embedding_cache,
                        )

                        optimizer.zero_grad()

                        with autocast(enabled=torch.cuda.is_available(), dtype=torch.bfloat16):
                            loss = neural_network.compute_loss(
                                source_batch,
                                target_batch,
                                source_kmer_embeddings=source_kmer_emb,
                                target_kmer_embeddings=target_kmer_emb,
                                index=i,
                            )

                        loss.backward()
                        optimizer.step()
                        scheduler.step()

                        total_loss += loss.item()
                        pbar.set_postfix({"loss": f"{loss.item():.4f}"})
                        pbar.update(1)

                avg_loss = total_loss / num_batches
                print(f"Epoch {epoch + 1}/{MaxEpoch}, Average Loss: {avg_loss:.4f}")

                # ====== 验证循环（与 train_lexicon.py 完全对齐）======
                NeuralNetwork = neural_network
                NeuralNetwork.eval()
                ValidProgressBar = tqdm(DevLoader)

                all_preds  = []
                all_labels = []
                all_logits = []

                with torch.no_grad():
                    for data in ValidProgressBar:
                        X, Y = data
                        X = list(X)
                        # 词汇嵌入
                        kmer_embeddings = match_and_embed_with_kmers_batch(
                            X, kmer_vocab, total_kmer_counts,
                            FastTextModel_target_path, target_kmer_embedding_cache
                        )
                        Y = Y.to(device)
                        output, attn_map = NeuralNetwork(X, Y, kmer_embeddings)
                        active_logits = output[0]
                        active_labels = output[1]
                        probabilities = F.softmax(active_logits, dim=-1)
                        preds = torch.argmax(probabilities, dim=-1)

                        all_preds.append(preds.cpu().numpy())
                        all_labels.append(active_labels.cpu().numpy())
                        all_logits.append(active_logits.cpu().numpy())

                all_preds  = np.concatenate(all_preds)
                all_labels = np.concatenate(all_labels)
                all_logits = np.concatenate(all_logits)

                acc, precision, recall, f1, mcc, auprc, jaccard = compute_metrics(
                    all_preds, all_labels, all_logits[:, 1], include_extra=True
                )
                print(f"Accuracy: {acc:.4f}, Precision: {precision:.4f}, Recall: {recall:.4f}, F1: {f1:.4f}, Mcc:{mcc:.4f}, AUPRC: {auprc:.4f}, Jaccard: {jaccard:.4f}")

                # 保存最优模型
                if mcc > best_mcc:
                    best_mcc = mcc
                    print(f"New best MCC: {best_mcc:.4f}, saving model...")
                    save_model_evosnr(
                        neural_network,
                        f"{save_dir}/best_evosnr_{target[:4]}_500bp_seed{seed}",
                    )

                # 早停判断
                early_stopping(mcc)
                if early_stopping.early_stop:
                    print(f"Early stopping triggered.Best MCC is{best_mcc}")
                    neural_network.to("cpu")
                    del neural_network
                    torch.cuda.empty_cache()
                    gc.collect()
                    break

        # ===========================================================
        # 测试模式
        # ===========================================================
        else:
            print("Enter the Test Mode --------------------->")
            seed_list = [1, 6, 8]
            results = []

            for seed in seed_list:
                set_seed(seed)
                load_dir_full = f"{save_dir}/best_evosnr_{target[:4]}_seed{seed}"

                neural_network = load_model_evosnr(
                    models.model_evosnr.EvoSNR,
                    load_dir_full,
                    device,
                )
                neural_network.eval()

                all_preds  = []
                all_labels = []
                all_logits = []

                TestProgressBar = tqdm(TestLoader)
                with torch.no_grad():
                    for data in TestProgressBar:
                        X, Y = data
                        X = list(X)
                        Y = Y.to(device)

                        kmer_emb_test = match_and_embed_with_kmers_batch(
                            X,
                            kmer_vocab,
                            total_kmer_counts,
                            FastTextModel_target_path,
                            target_kmer_embedding_cache,
                        )

                        output, attn_map = neural_network(X, Y, kmer_emb_test)
                        active_logits = output[0]
                        active_labels = output[1]

                        probabilities = F.softmax(active_logits.float(), dim=-1)
                        preds = torch.argmax(probabilities, dim=-1)

                        all_preds.append(preds.cpu().numpy())
                        all_labels.append(active_labels.cpu().numpy())
                        all_logits.append(active_logits.float().cpu().numpy())

                all_preds  = np.concatenate(all_preds)
                all_labels = np.concatenate(all_labels)
                all_logits = np.concatenate(all_logits)

                acc, precision, recall, f1, mcc, auprc, jaccard = compute_metrics(
                    all_preds, all_labels, all_logits[:, 1], include_extra=True
                )
                results.append({
                    "Seed":      seed,
                    "Accuracy":  acc,
                    "Precision": precision,
                    "Recall":    recall,
                    "F1":        f1,
                    "MCC":       mcc,
                    "AUPRC":     auprc,
                    "Jaccard":   jaccard,
                })
                print(f"[Seed {seed}] Acc:{acc:.4f} Prec:{precision:.4f} Rec:{recall:.4f} F1:{f1:.4f} MCC:{mcc:.4f} AUPRC:{auprc:.4f} Jaccard:{jaccard:.4f}")

                del neural_network
                torch.cuda.empty_cache()
                gc.collect()

            results_df = pd.DataFrame(results)
            result_csv = f"{save_dir}/../Experiment/results/evosnr_results({target[:4]}).csv"
            os.makedirs(os.path.dirname(result_csv), exist_ok=True)
            results_df.to_csv(
                result_csv,
                index=False,
                columns=["Seed", "Accuracy", "Precision", "Recall", "F1", "MCC", "AUPRC", "Jaccard"],
            )
            print(f"Results saved to {result_csv}")

    except KeyboardInterrupt:
        print("Training interrupted, releasing GPU memory...")
        try:
            del neural_network
        except NameError:
            pass
        torch.cuda.empty_cache()
        gc.collect()
        print("Memory released.")
        exit(0)


main()
