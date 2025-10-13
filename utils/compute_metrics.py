from sklearn.model_selection import train_test_split
import models.model_base
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score,matthews_corrcoef, classification_report
def compute_metrics(preds, labels, logits=None):
    # 计算 ACC, PRECISION, RECALL, F1
    acc = accuracy_score(labels, preds)
    precision = precision_score(labels, preds)  # weighted 为了多类情况
    recall=recall_score(labels, preds)
    f1 = f1_score(labels, preds)
    mcc = matthews_corrcoef(labels, preds)

    # report = classification_report(labels, preds, output_dict=True, zero_division=0)

    # 返回总体指标 和 Per-Class 指标（可以返回 report 字典）
    # 或者只返回总体指标，并在训练/评估流程中单独打印 report
    return acc,precision,recall,f1,mcc # 包含 Per-Class 指标
    

