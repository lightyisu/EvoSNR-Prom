#only Evo+SoftMAX
from tqdm import tqdm
import os
os.environ["HF_ENDPOINT"] = "https://alpha.hf-mirror.com"
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
from utils.compute_metrics import compute_metrics
from utils.loss import FocalLoss
from utils.seed import set_seed
from utils.save_load import save_model_complete,load_model_complete
from utils.early_stop import EarlyStopping
from utils.bash_config import parse_args
import models.model_base
import numpy as np
import torch
import pandas as pd  # Added for CSV handling
import gc
exec(open('configurator.py').read()) # 


args = parse_args()           



device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")



MaxEpoch = 60
BatchSize = 2
t = 0.15
ThresholdValue = 0.5
'''
    Initialization End    
'''







dev_path=PCA_split_path+'split_dev.csv'
train_path=PCA_split_path+'split_train.csv'
test_path=PCA_split_path+'split_test.csv'
print(f'loading : {train_path}')

LossFunction=FocalLoss(alpha=1, gamma=2)
# os.makedirs('ModelWeight/multiModel/' + TFsName)



TrainSequence, TrainDenseLabel=Datasets.DataReader_EVO.DataReaderBERT_SingleDomain(train_path)
DevSequence, DevDenseLabel=Datasets.DataReader_EVO.DataReaderBERT_SingleDomain(dev_path)
TestSequence, TestDenseLabel=Datasets.DataReader_EVO.DataReaderBERT_SingleDomain(test_path)


TrainSequence, TrainDenseLabel = \
    np.array(TrainSequence), np.array(TrainDenseLabel)

# TrainSequence, TestSequence, TrainDenseLabel, TestDenseLabel = train_test_split(TrainSequence, TrainDenseLabel, test_size=0.25, random_state=42)
LossFunction = LossFunction.to(device)



t_total = len(TrainSequence) * MaxEpoch // BatchSize


TrainDenseLabels = torch.tensor(TrainDenseLabel)
TrainLoader = Datasets.DataLoader.SampleLoaderBERT(data=TrainSequence, Label=TrainDenseLabels, BatchSize=BatchSize)

DevDenseLabels = torch.tensor(DevDenseLabel)
TestDenseLabels = torch.tensor(TestDenseLabel)
DevLoader = Datasets.DataLoader.SampleLoaderBERT(data=DevSequence, Label=DevDenseLabels,BatchSize=BatchSize)

TestLoader=Datasets.DataLoader.SampleLoaderBERT(data=TestSequence, Label=TestDenseLabels,BatchSize=BatchSize)



mode='train'
def main():
        try:
            
          
            if mode=='train':
                    seed = getattr(args, 'seed', 1)
                    set_seed(seed)
                   
                    NeuralNetwork = models.model_base.HyenaSegment().to(device)
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

                    optimizer = optim.AdamW(optimizer_grouped_parameters, lr=2e-4, eps=1e-8,
                                        betas=(0.9, 0.999))
                    scheduler = op.get_linear_schedule_with_warmup(
                    optimizer, num_warmup_steps=int(t_total * 0.1), num_training_steps=t_total
                    )
                    
                    print(f'Enter seed mode--------->Now Seed is :{seed}🌙')
                    early_stopping = EarlyStopping(patience=8) 
                   
                    print(f'Enter the Train Mode--------------------->')
                    best_mcc = -float('inf')  # 用于保存最佳MCC
                    
                    for Epoch in range(MaxEpoch):
                        # train
                        NeuralNetwork.train()
                        TrainProgressBar = tqdm(TrainLoader)
                        for data in TrainProgressBar:
                            optimizer.zero_grad()
                            
                            X, Y = data
                            X=list(X)
                            Y = Y.to(device)
                            output= NeuralNetwork(X,Y)
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
                        all_positive_probs = []  # 用于存储正类概率，供 AUPRC 计算

                        with torch.no_grad():
                            for data in ValidProgressBar:
                                X, Y = data
                                X=list(X)
                                Y = Y.to(device)
                                output= NeuralNetwork(X,Y)
                                active_logits=output[0]
                                active_labels=output[1]
                                probabilities = F.softmax(active_logits.float(), dim=-1)
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
                            all_preds, all_labels, all_positive_probs, include_extra=True
                        )

                        # 输出结果
                        print(
                            f"Accuracy: {acc:.4f}, Precision: {precision:.4f}, Recall: {recall:.4f}, "
                            f"F1: {f1:.4f}, Mcc:{mcc:.4f}, AUPRC: {auprc:.4f}, Jaccard: {jaccard:.4f}"
                        )
                        # 如果当前MCC比之前的最佳MCC好，则保存模型
                        if mcc > best_mcc:
                            best_mcc = mcc
                            
                            print(f"New best MCC: {best_mcc:.4f}")
                           
                            # ====== 修改：使用新的保存方法 ======
                            save_model_complete(NeuralNetwork, f'{save_dir}/best_evosnr(base)_{target[:4]}_seed{seed}')
                            # ====== 修改结束 ======
                        # 早停机制判断
                        early_stopping(mcc)  # 使用 MCC 作为监控指标
                        if early_stopping.early_stop :
                            print(f"Early stopping triggered.Best MCC is{best_mcc}")
                           
                          
                            break  # 提前终止训练    
                            
            else:
                print(f'Enter the Eval Mode--------------------->')
                seed_list=[1,6,8]
                results = []
                for seed in seed_list:  # Load the saved model
                    load_dir_full = f'{save_dir}/best_evosnr(base)_{target[:4]}_seed{seed}'
                    # 1）new 一个全新的 HyenaSegment 实例
                    NeuralNetwork = load_model_complete(
                        models.model_base.HyenaSegment,
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
                            output= NeuralNetwork(X,Y)
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
                            all_preds, all_labels, all_positive_probs, include_extra=True
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
                        f"F1: {f1:.4f}, Mcc:{mcc:.4f}, AUPRC: {auprc:.4f}, Jaccard: {jaccard:.4f}"
                    )
                    # 🔽🔽🔽 清理内存（重点） 🔽🔽🔽
                    del NeuralNetwork
                    torch.cuda.empty_cache()
                    gc.collect()
            # Convert results to DataFrame and save to CSV
                results_df = pd.DataFrame(results)
                results_df.to_csv(f'/root/autodl-tmp/zwk/evosnr_0605/Experiment/results/evosnr_base_results({target[:4]}).csv', index=False, columns=['Seed', 'Accuracy', 'Precision', 'Recall', 'F1', 'MCC', 'AUPRC', 'Jaccard'])
                print("Results saved to test_results.csv")
        except KeyboardInterrupt:
            print("Training interrupted, releasing GPU memory...")
            del NeuralNetwork  # 删除模型对象
            torch.cuda.empty_cache()  # 清理 PyTorch 缓存
            gc.collect()  # 强制运行垃圾回收
            print("Memory released.")
            exit(0)    
        

main()