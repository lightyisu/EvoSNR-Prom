import os
import gc
import logging
import torch
import torch.nn as nn
import numpy as np
from torch.nn import CrossEntropyLoss
from torch.utils.data import DataLoader
from transformers import AutoConfig
from peft import LoraConfig, get_peft_model, PeftModel
from evo import Evo
from evo.scoring import prepare_batch
from tqdm import tqdm
from sklearn.metrics import accuracy_score, f1_score, matthews_corrcoef, precision_score, recall_score
from torch.optim import AdamW

import Dataset.DataLoader
import Dataset.DataReader_EVO
import pandas as pd
from sklearn.model_selection import train_test_split  # 导入分割库

# 设置日志
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# 环境变量配置
os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "max_split_size_mb:32"

# 动态路径配置
BASE_DIR = os.getenv("MODEL_DIR", "/root/autodl-tmp/zwk/evosnr_0605/Experiment")
ADAPTER_PATH = os.path.join(BASE_DIR, "Baseline/slide_evo-1/model/adapters")
CLASSIFIER_PATH = os.path.join(BASE_DIR, "Baseline/slide_evo-1/model/classifier.pth")

PCA_DIR=os.path.join(BASE_DIR, "/root/autodl-tmp/zwk/evosnr_0605/data/Kleb/split")
# 设备配置
from sklearn.model_selection import KFold
device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")
logging.info(f"Using device: {device}")

class CustomEmbedding(nn.Module):
    def unembed(self, u):
        return u

class EvoPromoterClassifier(nn.Module):
    def __init__(self, pooling_strategy='mean', adapter_path=None, classifier_path=None):
        super(EvoPromoterClassifier, self).__init__()
        self.pooling_strategy = pooling_strategy
        self.num_labels = 2
        self.evo_dim = 512
        self.dropout = nn.Dropout(0.2)

        # 加载Evo模型
        logging.info("Loading Evo model: evo-1-8k-base")
        try:
            self.evo_model = Evo('evo-1-8k-base')
            self.bert = self.evo_model.model
            self.tokenizer = self.evo_model.tokenizer
        except Exception as e:
            logging.error(f"Failed to load Evo model: {e}")
            raise

        # 加载模型配置
        hf_model_name = 'togethercomputer/evo-1-8k-base'
        try:
            model_config = AutoConfig.from_pretrained(hf_model_name, trust_remote_code=True, revision='1.1_fix')
            self.bert.config = model_config
        except Exception as e:
            logging.error(f"Failed to load model config: {e}")
            raise

        # LoRA配置
        if adapter_path and os.path.exists(adapter_path):
            logging.info(f"Loading LoRA adapters from {adapter_path}")
            try:
                self.bert = PeftModel.from_pretrained(self.bert, adapter_path, is_trainable=False)
            except Exception as e:
                logging.error(f"Failed to load LoRA adapters: {e}")
                raise
        else:
            logging.info("Configuring new LoRA adapters")
            peft_config = LoraConfig(
                r=8,
                lora_alpha=32,
                lora_dropout=0.1,
                target_modules=[
                    "blocks.8.inner_mha_cls.Wqkv",
                    "blocks.8.inner_mha_cls.out_proj",
                    "blocks.16.inner_mha_cls.Wqkv",
                    "blocks.16.inner_mha_cls.out_proj",
                    "blocks.24.inner_mha_cls.Wqkv",
                    "blocks.24.inner_mha_cls.out_proj",
                ] + [f"blocks.{i}.mlp.l{j}" for i in range(32) for j in range(1, 4)]
            )
            self.bert = get_peft_model(self.bert, peft_config)
            self.bert.print_trainable_parameters()

        # 分类器
        self.classifier = nn.Linear(self.evo_dim, self.num_labels)
        if classifier_path and os.path.exists(classifier_path):
            logging.info(f"Loading classifier from {classifier_path}")
            try:
                self.classifier.load_state_dict(torch.load(classifier_path, map_location=device))
            except Exception as e:
                logging.error(f"Failed to load classifier: {e}")
                raise
        else:
            logging.warning("Classifier path not found, initializing new classifier")

        # 损失函数
        self.loss_fct = CrossEntropyLoss()

        # 移动到设备并转换为bfloat16
        self.bert = self.bert.to(device=device, dtype=torch.bfloat16)
        self.classifier = self.classifier.to(device=device, dtype=torch.bfloat16)

    def forward(self, input_seqs, label_ids=None, input_ids=None,attention_mask=None):
        self.bert.unembed = CustomEmbedding()
        
        if input_ids is not None:
            embed, _ = self.bert(input_ids)  # (batch, length, embed dim)
        else:
            # 现有逻辑处理字符串输入
            if isinstance(input_seqs, str):
                input_seqs = [input_seqs]
            try:
                input_ids, seq_lengths = prepare_batch(
                    input_seqs,
                    self.tokenizer,
                    prepend_bos=False,
                    device=device
                )
                embed, _ = self.bert(input_ids)
            except Exception as e:
                logging.error(f"Failed to prepare batch: {e}")
                raise

        sequence_output = self.dropout(embed)
        pooled_output = sequence_output.mean(dim=1)
        logits = self.classifier(pooled_output)

        loss = None
        if label_ids is not None:
            
            loss = self.loss_fct(logits.view(-1, self.num_labels), label_ids.to(logits.device).view(-1))
      
        return loss, logits
    



from ushuffle import shuffle

def shuffle_sequence(seq: str, k: int = 2) -> str:
    """
    使用 'ushuffle' Python库来打乱序列，同时保持k-mer频率。
    
    Args:
        seq (str): 要打乱的DNA字符串序列。
        k (int): 需要保持频率的k-mer长度（默认为2，即二核苷酸）。

    Returns:
        str: 打乱后的字符串序列。
    """
    # 1. 将输入的字符串(str)编码为字节(bytes)，因为ushuffle库需要字节作为输入
    sequence_bytes = seq.encode('utf-8')
    
    # 2. 调用ushuffle库的shuffle函数
    shuffled_bytes = shuffle(sequence_bytes, k)
    
    # 3. 将返回的字节(bytes)解码为字符串(str)以用于代码的其余部分
    shuffled_sequence_str = shuffled_bytes.decode('utf-8')
    
    return shuffled_sequence_str



def process_data_and_generate_negatives(sequences: list, labels: list) -> tuple:
    """
    将标签转换为序列级别，并为正样本生成打乱顺序的负样本。
    (此版本使用 ushuffle Python库)
    
    Args:
        sequences (list): DNA序列列表。
        labels (list): 每个序列对应的标签列表（每个碱基一个标签）。

    Returns:
        tuple: 包含新序列列表和新标签列表的元组。
    """
    new_sequences = []
    new_labels = []

    for seq, label_list in zip(sequences, labels):
        is_positive = 1 in label_list

        if is_positive:
            # 1. 添加原始正样本
            new_sequences.append(seq)
            new_labels.append(1)

            # 2. 调用新的 shuffle_sequence 函数生成负样本
            shuffled_seq = shuffle_sequence(seq, k=2) # 使用二核苷酸打乱
            new_sequences.append(shuffled_seq)
            new_labels.append(0)
        else:
            # 3. 添加原始负样本
            new_sequences.append(seq)
            new_labels.append(0)
            
    return new_sequences, new_labels

# 评估函数
def evaluate_model(model, val_loader):
    model.eval()
    all_preds, all_labels = [], []
    total_val_loss = 0

    with torch.no_grad():
        for batch in tqdm(val_loader, desc="Evaluating"):
            sequences, labels = batch
            try:
                loss, logits = model(input_seqs=sequences, label_ids=labels)
                total_val_loss += loss.item()
            except Exception as e:
                logging.error(f"Evaluation failed: {e}")
                continue

            probs = torch.softmax(logits, dim=-1)
            preds = torch.argmax(probs, dim=-1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    if not all_preds:
        logging.warning("No predictions made during evaluation")
        return float('inf'), 0, 0, 0, 0, 0

    acc = accuracy_score(all_labels, all_preds)
    f1 = f1_score(all_labels, all_preds, average='binary')
    mcc = matthews_corrcoef(all_labels, all_preds)
    precision = precision_score(all_labels, all_preds, average='binary')
    recall = recall_score(all_labels, all_preds, average='binary')
    avg_val_loss = total_val_loss / len(val_loader)

    return avg_val_loss, acc, f1, mcc, precision, recall

# 训练函数
def train_model(model, train_loader, val_loader, epochs=20, patience=10):
    optimizer = AdamW(model.parameters(), lr=5e-5)
    best_val_loss = float("inf")
    patience_counter = 0

    model.train()
    for epoch in range(epochs):
        total_train_loss = 0
        for batch in tqdm(train_loader, desc=f"Training Epoch {epoch+1}"):
            sequences, labels = batch
            optimizer.zero_grad()
            try:
                loss, _ = model(input_seqs=sequences, label_ids=labels)
                loss.backward()
                optimizer.step()
                total_train_loss += loss.item()
            except Exception as e:
                logging.error(f"Training step failed: {e}")
                continue

        avg_train_loss = total_train_loss / len(train_loader)
        val_loss, val_acc, val_f1, val_mcc, val_pre, val_rec = evaluate_model(model, val_loader)

        logging.info(f"Epoch {epoch+1}/{epochs}")
        logging.info(f"  Training Loss: {avg_train_loss:.4f}")
        logging.info(f"  Validation Loss: {val_loss:.4f}")
        logging.info(f"  Accuracy: {val_acc:.4f}, F1: {val_f1:.4f}, MCC: {val_mcc:.4f}")

        # 早停
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
            try:
                os.makedirs(ADAPTER_PATH, exist_ok=True)
                model.bert.save_pretrained(ADAPTER_PATH)
                torch.save(model.classifier.state_dict(), CLASSIFIER_PATH)
                logging.info("Model and classifier saved.")
            except Exception as e:
                logging.error(f"Failed to save model: {e}")
        else:
            patience_counter += 1
            if patience_counter >= patience:
                logging.info("Early stopping triggered.")
                break



def sliding_window_predict(model, sequence, window_size, device,batch_size=4):
  

    if window_size % 2 == 0:
        logging.warning("Window size should be odd, incrementing by 1")
        window_size += 1

    seq_len = len(sequence)
    half_window = window_size // 2
    predictions = np.zeros(seq_len, dtype=np.int32)  # 预分配预测结果数组

    # 生成所有窗口序列
    window_sequences = []
    for center_idx in range(seq_len):
        start_idx = max(0, center_idx - half_window)
        end_idx = min(seq_len, center_idx + half_window + 1)
        window_seq = sequence[start_idx:end_idx]

        # 处理边界填充
        if len(window_seq) < window_size:
            padding_needed = window_size - len(window_seq)
            if start_idx == 0:  # 左边界
                window_seq = window_seq + 'N' * padding_needed
            else:  # 右边界
                window_seq = 'N' * padding_needed + window_seq

        assert len(window_seq) == window_size, f"Window size mismatch: {len(window_seq)} != {window_size}"
        window_sequences.append(window_seq)

    # 批量推理
    model.eval()
    with torch.no_grad():
        for i in range(0, seq_len, batch_size):
            batch_seqs = window_sequences[i:i + batch_size]
            try:
                # 使用 prepare_batch 处理批量序列
                input_ids, seq_lengths = prepare_batch(
                    batch_seqs,
                    model.tokenizer,
                    prepend_bos=False,
                    device=device
                )
                _, logits = model(input_seqs=None, input_ids=input_ids)  # 直接使用 input_ids
                probs = torch.softmax(logits, dim=1)
                preds = torch.argmax(probs, dim=1).cpu().numpy()
                predictions[i:i + len(preds)] = preds
            except Exception as e:
                logging.error(f"Batch prediction failed at index {i}: {e}")
                predictions[i:i + batch_size] = 0  # 默认非启动子

    return predictions.tolist()

if __name__ == "__main__":
    # 清理显存

    metrics_dict = {
        'Fold': [],
        'Accuracy': [],
        'Precision': [],
        'Recall': [],
        'F1': [],
        'MCC': []
    }

   

    batch_size=8
   
    # model = EvoPromoterClassifier().to(device)
  
    
        # 1. 读取所有训练数据（原train_data.csv）
    #NOW DIRECTLY PCA INPUT CONVERT 2 (AUTO CLS-2)
    train_sequences, train_labels = Dataset.DataReader_EVO.DataReaderBERT_SingleDomain(
        os.path.join(PCA_DIR, "split_train.csv")
    )
    
    dev_sequences, dev_labels = Dataset.DataReader_EVO.DataReaderBERT_SingleDomain(
        os.path.join(PCA_DIR, "split_dev.csv")
    )

    #Original PCA input
    test_sequences, test_labels = Dataset.DataReader_EVO.DataReaderBERT_SingleDomain(
        os.path.join(PCA_DIR, "split_test.csv")
    )
    # 2. 处理训练和验证数据：转换标签并生成负样本
    logging.info("Processing train data for sequence-level classification...")
    train_sequences, train_labels = process_data_and_generate_negatives(train_sequences, train_labels)
   

    logging.info("Processing dev data for sequence-level classification...")
    dev_sequences, dev_labels = process_data_and_generate_negatives(dev_sequences, dev_labels)
    
    TrainLoader = Dataset.DataLoader.SampleLoaderBERT(
        data=train_sequences, Label=train_labels, BatchSize=batch_size
    )
    DevLoader = Dataset.DataLoader.SampleLoaderBERT(
        data=dev_sequences, Label=dev_labels, BatchSize=batch_size
    )
     
   
    logging.info("-------Entering train mode-------")
    # 取消注释以启用训练

    # train_model(model, TrainLoader, DevLoader, epochs=15)

    #     # 清理训练模型
    # del model
    # gc.collect()
    # torch.cuda.empty_cache()
    # logging.info(f"Memory after training cleanup: {torch.cuda.memory_allocated() / 1e6:.2f} MB")
    # except Exception as e:
    #     logging.error(f"Training setup failed: {e}")
    #     raise

    # 推理
    logging.info("-------Entering inference mode-------")
    kf = KFold(n_splits=5, shuffle=True)
    inference_model = EvoPromoterClassifier(
            adapter_path=ADAPTER_PATH,
            classifier_path=CLASSIFIER_PATH
        ).to(device)
    inference_model.eval()
    window_size = 81  # 滑动窗口大小
    batch_size = 4
    for fold, (train_idx, val_idx) in enumerate(kf.split(test_sequences), 1):
                logging.info(f"-------运行第 {fold}/5 折-------")
                print(f"\n运行第 {fold} 折实验")
                val_sequences = [test_sequences[i] for i in val_idx]
                val_labels = [test_labels[i] for i in val_idx]
              
                
                # 批量推理
                
                all_preds = []
                for seq in tqdm(val_sequences, desc=f"Fold {fold} Prediction"):
                        pred = sliding_window_predict(inference_model, seq, window_size, device, batch_size)
                        all_preds.append(pred)
              
              
                if len(all_preds) != len(val_labels):
                        logging.error(f"预测列表长度 {len(all_preds)} 与标签长度 {len(val_labels)} 不匹配")
                        continue
            # # 运行滑动窗口分类
                # classifications = sliding_window_predict(seq, model, WINDOW_SIZE)
   
    

                y_pred_flat = np.concatenate([np.array(seq) for seq in all_preds])
                y_true_flat = np.concatenate([np.array(seq) for seq in val_labels])

                # 计算指标
                accuracy = accuracy_score(y_true_flat, y_pred_flat)
                precision = precision_score(y_true_flat, y_pred_flat, zero_division=0)
                recall = recall_score(y_true_flat, y_pred_flat, zero_division=0)
                f1 = f1_score(y_true_flat, y_pred_flat, zero_division=0)
                mcc = matthews_corrcoef(y_true_flat, y_pred_flat)

                # 保存结果
                metrics_dict['Fold'].append(fold)
                metrics_dict['Accuracy'].append(accuracy)
                metrics_dict['Precision'].append(precision)
                metrics_dict['Recall'].append(recall)
                metrics_dict['F1'].append(f1)
                metrics_dict['MCC'].append(mcc)

                # 打印结果
                print(f"折 {fold} - 准确率: {accuracy:.4f}")
                print(f"折 {fold} - 精确率: {precision:.4f}")
                print(f"折 {fold} - 召回率: {recall:.4f}")
                print(f"折 {fold} - F1 分数: {f1:.4f}")
                print(f"折 {fold} - MCC: {mcc:.4f}")

    # 保存结果到 CSV 文件
    results_df = pd.DataFrame(metrics_dict)
    output_path = os.path.join(BASE_DIR, "results", "evo_slide(KLEB).csv")
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    results_df.to_csv(output_path, index=False)
    print(f"\n结果已保存至 {output_path}")
   
    # 清理推理模型
    del inference_model
    gc.collect()
    torch.cuda.empty_cache()
    logging.info(f"Memory after cleanup: {torch.cuda.memory_allocated() / 1e6:.2f} MB")

   