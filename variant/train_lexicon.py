
from tqdm import tqdm
import os
os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"
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
# need model_lex.py
import models.model_lex
#词汇信息预处理
import numpy as np
from collections import defaultdict
import gc





from utils.loss import FocalLoss
from utils.compute_metrics import compute_metrics
from utils.seed import set_seed
from utils.save_load import save_model_lexicon,load_model_lexicon
from utils.early_stop import EarlyStopping
from utils.count_lexicon import countLexiconKmer
from utils.lexicon import count_kmers_in_dataset_with_vocab,readLexicon,match_and_embed_with_kmers_batch

from utils.fasttext_load import get_direct_vector
from utils.bash_config import parse_args



exec(open('configurator.py').read()) # overrides from command line or config file
kmer_vocab=readLexicon(Lexicon_target_path)

# 打印词汇表大小和嵌入维度
print(f"词汇表大小: {len(kmer_vocab)}")

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")
args = parse_args()     
MaxEpoch = 30
BatchSize = 2
t = 0.15
ThresholdValue = 0.5


dev_path=PCA_split_path+'split_dev.csv'
train_path=PCA_split_path+'split_train.csv'
test_path=PCA_split_path+'split_test.csv'


LossFunction=FocalLoss(alpha=1, gamma=2)
LossFunction = LossFunction.to(device)


TrainSequence, TrainDenseLabel = \
    Datasets.DataReader_EVO.DataReaderBERT_SingleDomain(train_path)
DevSequence, DevDenseLabel=Datasets.DataReader_EVO.DataReaderBERT_SingleDomain(dev_path)
TestSequence, TestDenseLabel=Datasets.DataReader_EVO.DataReaderBERT_SingleDomain(test_path)


TrainSequence, TrainDenseLabel = \
    np.array(TrainSequence), np.array(TrainDenseLabel)


TrainDenseLabels = torch.tensor(TrainDenseLabel)
TrainLoader = Datasets.DataLoader.SampleLoaderBERT(data=TrainSequence, Label=TrainDenseLabels, BatchSize=BatchSize)

DevDenseLabels = torch.tensor(DevDenseLabel)
DevLoader = Datasets.DataLoader.SampleLoaderBERT(data=DevSequence, Label=DevDenseLabels,BatchSize=BatchSize)

TestDenseLabels = torch.tensor(TestDenseLabel)
TestLoader=Datasets.DataLoader.SampleLoaderBERT(data=TestSequence, Label=TestDenseLabels,BatchSize=BatchSize)


# 统计k-mer频率
total_kmer_counts = defaultdict(int)
TrainProgressBar = tqdm(TrainLoader)
DevProgressBar=tqdm(DevLoader)
TestProgressBar=tqdm(TestLoader)

#训练集测试集频率统计
total_kmer_counts=countLexiconKmer(TrainProgressBar,kmer_vocab,total_kmer_counts)
total_kmer_counts=countLexiconKmer(TestProgressBar,kmer_vocab,total_kmer_counts)

target_kmer_embedding_cache = {kmer: get_direct_vector(kmer, FastTextModel_target_path) for kmer in kmer_vocab if kmer in total_kmer_counts}



mode='train'
def main():
    try:
      
       
        if mode=='train':
        
            print(f'Enter the Train Mode--------------------->')
            seed = getattr(args, 'seed', 1)
            set_seed(seed)
            NeuralNetwork = models.model_lex.HyenaSegment().to(device)

            t_total = len(TrainSequence) * MaxEpoch // BatchSize

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

            # optimizer = optim.AdamW(optimizer_grouped_parameters, lr=2e-4, eps=1e-8,
            #                     betas=(0.9, 0.999))
            evo_params = list(map(id, NeuralNetwork.evo.parameters()))
            # downstream_params = filter(lambda p: id(p) not in evo_params, NeuralNetwork.parameters())
             # 门控融合模块的参数单独分组，给予中等学习率
            fusion_params = list(map(id, NeuralNetwork.gated_fusion.parameters()))
            downstream_params = filter(
                lambda p: id(p) not in evo_params and id(p) not in fusion_params, 
                NeuralNetwork.parameters()
            )


            optimizer_grouped_parameters = [
             {"params": NeuralNetwork.evo.parameters(), "lr": 5e-5},           # EVO: 小学习率
                {"params": NeuralNetwork.gated_fusion.parameters(), "lr": 1e-4},   # 门控融合: 中等学习率
                {"params": downstream_params, "lr": 2e-4}                          # 下游模块: 大学习率
            ]

            # 注意：这里为了简化，没有再区分 weight_decay，您可以按需合并
            optimizer = optim.AdamW(optimizer_grouped_parameters, eps=1e-8, betas=(0.9, 0.999))
            scheduler = op.get_linear_schedule_with_warmup(
            optimizer, num_warmup_steps=int(t_total * 0.1), num_training_steps=t_total
            )
         
            print(f'Enter seed mode--------->Now Seed is :{seed}🌙')
            early_stopping = EarlyStopping(patience=15) 
                
            best_mcc = -float('inf')  # 用于保存最佳MCC
        
            for Epoch in range(MaxEpoch):
                # train
                NeuralNetwork.train()
                TrainProgressBar = tqdm(TrainLoader)
                for data in TrainProgressBar:
                    optimizer.zero_grad()
                    
                    X, Y = data
                    X=list(X)
                    
                    kmer_embeddings=match_and_embed_with_kmers_batch(X, kmer_vocab,total_kmer_counts,FastTextModel_target_path,target_kmer_embedding_cache)

                    Y = Y.to(device)
                    output,attn_map= NeuralNetwork(X,Y,kmer_embeddings)
                    active_logits=output[0]
                    active_labels=output[1]
                    #ac_log 179 ac_lab 180
                    Loss = LossFunction(active_logits, active_labels)

                
                
                    TrainProgressBar.set_description(f'Epoch {Epoch} loss:{Loss.cpu().item()}')
                    Loss.backward()
                    optimizer.step()
                    scheduler.step()
                # valid
                NeuralNetwork.eval()
                ValidProgressBar = tqdm(DevLoader)

                all_preds = []
                all_labels = []
                all_positive_probs = []
                
                with torch.no_grad():
                    for data in ValidProgressBar:
                        X, Y = data
                        X=list(X)
                        #加入词汇嵌入
                        kmer_embeddings=match_and_embed_with_kmers_batch(X, kmer_vocab,total_kmer_counts,FastTextModel_target_path,target_kmer_embedding_cache)
                        Y = Y.to(device)
                        output,attn_map= NeuralNetwork(X,Y,kmer_embeddings)
                        active_logits=output[0]
                        active_labels=output[1]
                        probabilities = F.softmax(active_logits, dim=-1)
                        preds= torch.argmax(probabilities, dim=-1)
                        
                        all_preds.append(preds.cpu().numpy())
                        all_labels.append(active_labels.cpu().numpy())
                        all_positive_probs.append(probabilities[:, 1].cpu().numpy())
                # 将预测、标签和正类概率合并为一个 numpy 数组
                all_preds = np.concatenate(all_preds)
                all_labels = np.concatenate(all_labels)
                all_positive_probs = np.concatenate(all_positive_probs)
                # 计算评估指标
                acc, precision, recall, f1, mcc, auprc, jaccard = compute_metrics(
                        all_preds,
                        all_labels,
                        all_positive_probs,
                        include_extra=True,
                    )

                # 输出结果
                print(
                    f"Accuracy: {acc:.4f}, Precision: {precision:.4f}, Recall: {recall:.4f}, "
                    f"F1: {f1:.4f}, MCC: {mcc:.4f}, AUPRC: {auprc:.4f}, Jaccard: {jaccard:.4f}"
                )
                # 如果当前MCC比之前的最佳MCC好，则保存模型
                if mcc > best_mcc:
                    best_mcc = mcc
                    print(f"New best MCC: {best_mcc:.4f}, saving the model...")
            
                    save_model_lexicon(NeuralNetwork, f'{save_dir}/best_evosnr(lexicon)_{target[:4]}_seed{seed}')
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
       
                     

        else:
            
            

            print(f'Enter the Eval Mode--------------------->')
            seed_list=[1,6,8]
            results = []
            for seed in seed_list:  # Load the saved model
                    set_seed(seed)
                    load_dir_full = f'{save_dir}/best_evosnr(lexicon)_{target[:4]}_seed{seed}'
                    # 1）new 一个全新的 HyenaSegment 实例
                    NeuralNetwork = load_model_lexicon(
                        models.model_lex.HyenaSegment,
                        load_dir_full,
                        device
                    )
                    NeuralNetwork.eval()

                    
            
                    all_preds = []
                    all_labels = []
                    all_positive_probs = []
                    TestProgressBar = tqdm(TestLoader)
                    
                    with torch.no_grad():
                        for data in TestProgressBar:
                            
                            X, Y = data
                            X=list(X)
                            Y = Y.to(device)
                            kmer_embeddings=match_and_embed_with_kmers_batch(X, kmer_vocab,total_kmer_counts,FastTextModel_target_path,target_kmer_embedding_cache)
                            output,attn_map= NeuralNetwork(X,Y,kmer_embeddings)
                            active_logits=output[0]
                            active_labels=output[1]
                            probabilities = F.softmax(active_logits, dim=-1)
                            preds= torch.argmax(probabilities, dim=-1)
                            
                            all_preds.append(preds.cpu().numpy())
                            all_labels.append(active_labels.cpu().numpy())
                            all_positive_probs.append(probabilities[:, 1].cpu().numpy())
                    # 将预测、标签和正类概率合并为一个 numpy 数组
                    all_preds = np.concatenate(all_preds)
                    all_labels = np.concatenate(all_labels)
                    all_positive_probs = np.concatenate(all_positive_probs)
                    # 计算评估指标
                    acc, precision, recall, f1, mcc, auprc, jaccard = compute_metrics(
                        all_preds,
                        all_labels,
                        all_positive_probs,
                        include_extra=True,
                    )
                    # Store metrics for this fold
                    results.append({
                        'Seed': seed,
                        'Accuracy': acc,
                        'Precision': precision,
                        'Recall': recall,
                        'F1': f1,
                        'MCC': mcc,
                        'AUPRC': auprc,
                        'Jaccard': jaccard
                    })  
                    # 输出结果
                    print(
                        f"Accuracy: {acc:.4f}, Precision: {precision:.4f}, Recall: {recall:.4f}, "
                        f"F1: {f1:.4f}, MCC: {mcc:.4f}, AUPRC: {auprc:.4f}, Jaccard: {jaccard:.4f}"
                    )
                    # 🔽🔽🔽 清理内存（重点） 🔽🔽🔽
                    del NeuralNetwork
                    torch.cuda.empty_cache()
                    gc.collect()
        # Convert results to DataFrame and save to CSV
            results_df = pd.DataFrame(results)
            results_df.to_csv(
                f'/root/autodl-tmp/zwk/evosnr_0605/Experiment/results/evosnr_lexicon_results({target[:4]}).csv',
                index=False,
                columns=['Seed', 'Accuracy', 'Precision', 'Recall', 'F1', 'MCC', 'AUPRC', 'Jaccard'],
            )
            print("Results saved to test_results.csv")
    except KeyboardInterrupt:
        print("Training interrupted, releasing GPU memory...")
        del NeuralNetwork  # 删除模型对象
        torch.cuda.empty_cache()  # 清理 PyTorch 缓存
        gc.collect()  # 强制运行垃圾回收
        print("Memory released.")
        exit(0)       
    

main()