#词汇信息预处理
import numpy as np
from collections import defaultdict
import gc
from torch.utils.data import Dataset, DataLoader, Sampler
from utils.compute_metrics import compute_metrics
from utils.loss import FocalLoss
from utils.seed import set_seed
from utils.save_load import save_model_la_dtl,load_model_la_dtl
from utils.early_stop import EarlyStopping
from utils.count_lexicon import countLexiconKmer
from utils.lexicon import readLexicon,match_and_embed_with_kmers_batch
from utils.bash_config import parse_args
from utils.fasttext_load import get_direct_vector
from torch.cuda.amp import autocast, GradScaler
from tqdm import tqdm
import os
# os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import transformers.optimization as op
import pandas as pd
import os
import torch.nn.functional as F
import Datasets.DataLoader
import Datasets.DataReader_EVO
import matplotlib.pyplot as plt

import models.model_la_dtl
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score,matthews_corrcoef


exec(open('configurator.py').read()) # overrides from command line or config file
args = parse_args()    

kmer_vocab_source=readLexicon(Lexicon_source_path)
kmer_vocab_target=readLexicon(Lexicon_target_path)
# 打印词汇表大小和嵌入维度
print(f"source词汇表大小: {len(kmer_vocab_source)}")
print(f"target词汇表大小: {len(kmer_vocab_target)}")




device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

MaxEpoch = 60
BatchSize = 4
t = 0.15
ThresholdValue = 0.5
'''
    Initialization End    
'''

LossFunction=FocalLoss(alpha=1, gamma=2)



#重复平衡采样器
class DualRepeatBalancedBatchSampler(Sampler):
    def __init__(self, source_dataset, target_dataset, batch_size):
        """
        改进的平衡采样器 - 完全利用源域，智能重复目标域
        
        参数:
            source_dataset: 源域数据集(通常较大)
            target_dataset: 目标域数据集(通常较小)
            batch_size: 总批次大小(必须为偶数)
        """
        self.source_dataset = source_dataset
        self.target_dataset = target_dataset
        self.batch_size = batch_size
        
        # 验证批次大小
        if batch_size % 2 != 0:
            raise ValueError("批次大小必须是偶数，以确保源域和目标域平衡")
        
        self.source_len = len(source_dataset)
        self.target_len = len(target_dataset)
        
        # 计算源域完整遍历所需的批次数
        self.num_batches = int(np.ceil(self.source_len / (batch_size // 2)))
        
        # 计算目标域需要重复的次数
        self.target_repeats = max(1, int(np.ceil(
            (self.num_batches * (batch_size // 2)) / self.target_len
        )))

    def __iter__(self):
        # 步骤1: 准备源域索引(完整遍历)
        source_indices = torch.randperm(self.source_len).tolist()
        
        # 步骤2: 准备目标域索引(重复采样)
        # 创建扩展的目标域索引列表，使其长度 >= 源域索引长度
        target_indices = []
        for _ in range(self.target_repeats):
            target_indices.extend(torch.randperm(self.target_len).tolist())
        target_indices = target_indices[:self.num_batches * (self.batch_size // 2)]
        
        # 步骤3: 生成平衡批次
        for i in range(self.num_batches):
            # 源域批次(可能包含填充)
            s_start = i * (self.batch_size // 2)
            s_end = s_start + (self.batch_size // 2)
            s_batch = source_indices[s_start:s_end]
            
            # 处理源域末尾不足的情况(循环填充)
            if len(s_batch) < self.batch_size // 2:
                needed = self.batch_size // 2 - len(s_batch)
                s_batch += source_indices[:needed]
            
            # 目标域批次
            t_start = i * (self.batch_size // 2)
            t_end = t_start + (self.batch_size // 2)
            t_batch = target_indices[t_start:t_end]
            
            yield s_batch, t_batch

    def __len__(self):
        return self.num_batches


      

#平衡的采样器
#Design For EVO-LaDTL Model
class DualBalancedBatchSampler(Sampler):
    def __init__(self, source_dataset, target_dataset, batch_size):
        self.source_dataset = source_dataset
        self.target_dataset = target_dataset
        self.batch_size = batch_size
        self.source_len = len(source_dataset)
        self.target_len = len(target_dataset)
        self.num_batches = min(self.source_len, self.target_len) // (self.batch_size // 2)

    def __iter__(self):
        # 随机打乱索引
        source_indices = torch.randperm(self.source_len).tolist()
        target_indices = torch.randperm(self.target_len).tolist()

        for i in range(self.num_batches):
            start = i * (self.batch_size // 2)
            end = start + (self.batch_size // 2)
            s_batch = source_indices[start:end]
            t_batch = target_indices[start:end]
            # 如果数量不足，则补充
            if len(s_batch) < self.batch_size // 2:
                s_batch += source_indices[:self.batch_size // 2 - len(s_batch)]
            if len(t_batch) < self.batch_size // 2:
                t_batch += target_indices[:self.batch_size // 2 - len(t_batch)]
            yield s_batch, t_batch

    def __len__(self):
        return self.num_batches



# ### 2. 定义自定义 Dataset
class NERDataset(Dataset):
    def __init__(self, sentences, tags):
        self.sentences = sentences
        self.tags = tags

    def __len__(self):
        return len(self.sentences)

    def __getitem__(self, idx):
        return self.sentences[idx], self.tags[idx]
    


#load dataset NOW

source_path=Source_PCA_path
dev_path=PCA_split_path+'split_dev.csv'
train_path=PCA_split_path+'split_train.csv'
test_path=PCA_split_path+'split_test.csv'


#-------datasets-------
SourceSequenceRaw, SourceLabelRaw,  TargetTrainSequenceRaw, TargetTrainLabelRaw = \
    Datasets.DataReader_EVO.DataReaderBERT(path1=source_path,path2=train_path)
SourceSequence, SourceLabel, TargetTrainSequence, TargetTrainLabel = \
    np.array(SourceSequenceRaw), np.array(SourceLabelRaw), \
    np.array(TargetTrainSequenceRaw), np.array(TargetTrainLabelRaw)

TargetDevSequenceRaw, TargetDevLabelRaw,  TargetTestSequenceRaw, TargetTestLabelRaw = \
    Datasets.DataReader_EVO.DataReaderBERT(path1=dev_path,path2=test_path)

TargetDevSequence, TargetDevLabel, TargetTestSequence, TargetTestLabel = \
    np.array(TargetDevSequenceRaw), np.array(TargetDevLabelRaw), \
    np.array(TargetTestSequenceRaw), np.array(TargetTestLabelRaw)

TargetDevLabelRaw=torch.tensor(TargetDevLabelRaw).to(device)
#---- end ------

SourceLoader = Datasets.DataLoader.SampleLoaderBERT(data=SourceSequence, Label=SourceLabel, BatchSize=BatchSize)

TargetTrainLoader = Datasets.DataLoader.SampleLoaderBERT(data=TargetTrainSequence, Label=TargetTrainLabel,BatchSize=BatchSize)

TargetDevLoader = Datasets.DataLoader.SampleLoaderBERT(data=TargetDevSequence, Label=TargetDevLabel,BatchSize=BatchSize)

TargetTestLoader = Datasets.DataLoader.SampleLoaderBERT(data=TargetTestSequence, Label=TargetTestLabel,BatchSize=BatchSize)
#-----end---
# 统计k-mer频率

source_kmer_counts = defaultdict(int)
target_kmer_counts=defaultdict(int)

SourceProgressBar = tqdm(SourceLoader)
TargetProgressBar=tqdm(TargetTrainLoader)
TargetDevProgressBar=tqdm(TargetDevLoader)
TargetTestProgressBar=tqdm(TargetTestLoader)

source_kmer_counts=countLexiconKmer(SourceProgressBar,kmer_vocab_source,source_kmer_counts)
target_kmer_counts=countLexiconKmer(TargetProgressBar,kmer_vocab_target,target_kmer_counts)
target_kmer_counts=countLexiconKmer(TargetDevProgressBar,kmer_vocab_target,target_kmer_counts)
target_kmer_counts=countLexiconKmer(TargetTestProgressBar,kmer_vocab_target,target_kmer_counts)
print('cache loading----')
source_kmer_embedding_cache = {kmer: get_direct_vector(kmer, FastTextModel_source_path) for kmer in kmer_vocab_source if kmer in source_kmer_counts}
target_kmer_embedding_cache = {kmer: get_direct_vector(kmer, FastTextModel_target_path) for kmer in kmer_vocab_target if kmer in target_kmer_counts}
mode='train'
print('cache loaded----')

        

def main():
    if mode=='train':

      # La-DTL VER- Load and split data
       #source | target(train,validation,test)
        print(f'Enter the Train Mode--------------------->')
        seed = getattr(args, 'seed', 1)
        set_seed(seed)
        NeuralNetwork = models.model_la_dtl.EvoSegmentLaDTL().to(device)
     
        

        
        t_total = len(SourceSequence) * MaxEpoch // BatchSize

        no_decay = ["bias", "LayerNorm.weight"]
        optimizer_grouped_parameters = [
        {
            "params": [p for n, p in NeuralNetwork.named_parameters() if not any(nd in n for nd in no_decay)],
            "weight_decay": 0.01,
        },
        {
            "params": [p for n, p in NeuralNetwork.named_parameters() if any(nd in n for nd in no_decay)],
            "weight_decay": 0.0,
        },
        ]
        evo_params = list(map(id, NeuralNetwork.evo.parameters()))
        downstream_params = filter(lambda p: id(p) not in evo_params, NeuralNetwork.parameters())

        optimizer_grouped_parameters = [
            {"params": NeuralNetwork.evo.parameters(), "lr": 5e-5},  # 为EVO设置一个较小的学习率
            {"params": downstream_params, "lr": 2e-4}  # 为下游任务模块设置一个较大的学习率
        ]

        # 注意：这里为了简化，没有再区分 weight_decay
        optimizer = optim.AdamW(optimizer_grouped_parameters, eps=1e-8, betas=(0.9, 0.999))
        # optimizer = optim.AdamW(optimizer_grouped_parameters, lr=2e-4, eps=1e-8,
        #                     betas=(0.9, 0.999))
        scheduler = op.get_linear_schedule_with_warmup(
        optimizer, num_warmup_steps=int(t_total * 0.1), num_training_steps=t_total
        )
        #---------------------------------------
        print(f'Enter seed mode--------->Now Seed is :{seed}🌙')
        early_stopping = EarlyStopping(patience=15) 
                
        best_mcc = -float('inf')


        
        source_dataset = NERDataset(SourceSequenceRaw, SourceLabelRaw)
        target_dataset = NERDataset(TargetTrainSequenceRaw, TargetTrainLabelRaw)
        target_val_dataset = NERDataset(TargetDevSequenceRaw, TargetDevLabelRaw)
       
        # 创建平衡采样器
        # sampler = DualBalancedBatchSampler(source_dataset, target_dataset, BatchSize)
        sampler=DualRepeatBalancedBatchSampler(source_dataset, target_dataset, BatchSize)
        #创建重复平衡采样器
       
        # 创建验证集DataLoader
        Val_loader = DataLoader(target_val_dataset, batch_size=BatchSize, shuffle=True)
        

        print(f'Enter the Train Mode--------------------->')
       
        best_mcc = -float('inf')  # 用于保存最佳MCC
        # save_dir = '/data/zwk/save_models/Evo'  # 定义模型保存路径



  
        for Epoch in range(MaxEpoch):
            # train
            NeuralNetwork.train()
            # TrainProgressBar = tqdm(TrainLoader)
            total_loss = 0.0
            #balance sample batcher
            num_batches = len(sampler)
            with tqdm(total=num_batches, desc=f"Epoch {Epoch+1}/{MaxEpoch}", unit="batch") as pbar:
                # 直接从数据集中取数据
                for i, (s_indices, t_indices) in enumerate(sampler):
                       
                        source_batch = [source_dataset[idx] for idx in s_indices]
                        target_batch = [target_dataset[idx] for idx in t_indices]

                    # for data in TrainProgressBar:
                        optimizer.zero_grad()
                        kmer_embeddings_X_source=[ item[0] for item in source_batch]
                        kmer_embeddings_X_target=[ item[0] for item in target_batch]
                         #加入词汇嵌入
                        kmer_embeddings_source=match_and_embed_with_kmers_batch(kmer_embeddings_X_source, kmer_vocab_source,source_kmer_counts,FastTextModel_source_path,source_kmer_embedding_cache)
                        kmer_embeddings_target=match_and_embed_with_kmers_batch(kmer_embeddings_X_target, kmer_vocab_target,target_kmer_counts,FastTextModel_target_path,target_kmer_embedding_cache)
                        
                        ##La-DTL LOSS COMPUTE
                        optimizer.zero_grad()
                    
                        loss = NeuralNetwork.compute_loss(source_batch, target_batch,kmer_embeddings_source,kmer_embeddings_target, index=i)
                       
                        loss.backward()
                        optimizer.step()
                        scheduler.step()
                        total_loss += loss.item()
                        pbar.set_postfix({'loss': f"{loss.item():.4f}"})
                        pbar.update(1)
                        
                        if torch.cuda.is_available():
                            torch.cuda.empty_cache()

            avg_loss = total_loss / num_batches  
            print(f"Epoch {Epoch+1}/{MaxEpoch}, Average Loss: {avg_loss:.4f}")




            # Evaluate on validation set
            NeuralNetwork.eval()
            val_preds = []
            val_labels = []
            total_val_loss = 0.0

            with torch.no_grad():
                for val_batch in tqdm(Val_loader, desc=f"Epoch {Epoch+1}/{MaxEpoch} - Validation"):
                    val_sentences,val_tags = val_batch
                    kmer_embeddings_val = match_and_embed_with_kmers_batch(val_sentences, kmer_vocab_target, target_kmer_counts,FastTextModel_target_path,target_kmer_embedding_cache)
                    
                    
                  
                    logits = NeuralNetwork.hidden2tag_t(
                        NeuralNetwork.EvoEmb(val_sentences, kmer_embeddings_val)
                    )
                    labels_tensor = val_tags.to(device)  # [B, L]
                    logits_flat = logits.view(-1, NeuralNetwork.num_labels)  # [B*L, C]
                    labels_flat = labels_tensor.view(-1)  # [B*L]
                    loss = NeuralNetwork.loss_fn(logits_flat, labels_flat) 
                    
                    
                    total_val_loss += loss.item()
                    pred_tags = torch.argmax(logits, dim=-1)
                    pred_tags = pred_tags.cpu().numpy()  
                    # 将预测和标签展平
                    for pred_seq, true_seq in zip(pred_tags, val_tags.cpu().numpy()):
                        val_preds.extend(pred_seq)
                        val_labels.extend(true_seq[:len(pred_seq)])  # 对齐序列长度

            avg_val_loss = total_val_loss / len(Val_loader)
            acc, precision, recall, f1, mcc = compute_metrics(val_preds, val_labels)
            print(f"Validation Results - Epoch {Epoch+1}/{MaxEpoch}:")
            print(f"Average Validation Loss: {avg_val_loss:.4f}")
            print(f"Accuracy: {acc:.4f}")
            print(f"Precision: {precision:.4f}")
            print(f"Recall: {recall:.4f}")
            print(f"F1 Score: {f1:.4f}")
            print(f"MCC: {mcc:.4f}")
            print("-" * 50)
            if mcc > best_mcc:
                best_mcc = mcc
                print(f"New best MCC: {best_mcc:.4f}, saving the model...")
            
                save_model_la_dtl(NeuralNetwork, f'{save_dir}/best_evosnr(la-dtl)_{target[:4]}_repeat_seed{seed}')
    
             # 早停机制判断
            early_stopping(mcc)  # 使用 MCC 作为监控指标
            if early_stopping.early_stop:
                print(f"Early stopping triggered.Best MCC is{best_mcc}")
                NeuralNetwork.to('cpu')
                # 🔽🔽🔽 清理内存（重点） 🔽🔽🔽
                del NeuralNetwork
                torch.cuda.empty_cache()
                gc.collect()
                break  # 提前终止训练    
    
    elif mode=='test':
        print(f'Enter the Test Mode--------------------->')     
        seed_list=[1,6,8]
        results = []   
        positive_predicted_sequences = []
        for seed in seed_list: 
            load_dir_full = f'{save_dir}/best_evosnr(la-dtl)_{target[:4]}_repeat_seed{seed}'
                   
            NeuralNetwork = load_model_la_dtl(
               models.model_la_dtl.EvoSegmentLaDTL,
                load_dir_full,
                device
            )
            NeuralNetwork.eval()
        
        
         # 进行测试
       
            all_preds = []
            all_labels = []
            all_logits = []  
            TestProgressBar = tqdm(TargetTestLoader)
            with torch.no_grad():
                  for data in TestProgressBar:
                    test_seqs, test_tags = data
                    # X=list(X)
                    # Y = Y.to(device)
                    #
                    
                    # 获取k-mer嵌入
                    kmer_embeddings = match_and_embed_with_kmers_batch(
                        test_seqs, kmer_vocab_target, target_kmer_counts, FastTextModel_target_path,target_kmer_embedding_cache
                    )
                    
                   
                    logits = NeuralNetwork.hidden2tag_t(
                        NeuralNetwork.EvoEmb(test_seqs, kmer_embeddings)
                    )
                    labels_tensor = test_tags.to(device)  # [B, L]
                    logits_flat = logits.view(-1, NeuralNetwork.num_labels)  # [B*L, C]
                    labels_flat = labels_tensor.view(-1)  # [B*L]
                    pred_tags = torch.argmax(logits, dim=-1)
                    pred_tags = pred_tags.cpu().numpy()  
                    
                    
                    
                    for pred_seq, true_seq in zip(pred_tags, test_tags.cpu().numpy()):
                        all_preds.extend(pred_seq)
                        all_labels.extend(true_seq[:len(pred_seq)])  # 对齐序列长度
            
            # 计算评估指标
            acc, precision, recall, f1, mcc = compute_metrics(all_preds, all_labels)
            results.append({
                    'Seed': seed,
                    'Accuracy': acc,
                    'Precision': precision,
                    'Recall': recall,
                    'F1': f1,
                    'MCC': mcc
                })  
            # 输出结果
            print(f"Accuracy: {acc:.4f}, Precision: {precision:.4f},F1: {f1:.4f}, Mcc:{mcc:.4f}")
            # 🔽🔽🔽 清理内存（重点） 🔽🔽🔽
            del NeuralNetwork
            torch.cuda.empty_cache()
            gc.collect()


        results_df = pd.DataFrame(results)
        results_df.to_csv(f'/root/autodl-tmp/zwk/evosnr_0605/Experiment/results/evosnr_la-dtl_results({target[:4]}).csv', index=False, columns=['Seed', 'Accuracy', 'Precision', 'Recall', 'F1', 'MCC'])
        print("Results saved to test_results.csv")
main()
