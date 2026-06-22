import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from peft import PeftModel
from tqdm import tqdm
import os

def save_model_la_dtl(model: nn.Module, save_path: str):

  # 创建保存目录
    os.makedirs(save_path, exist_ok=True)

    # 1. 保存 LoRA adapter
    model.evo.save_pretrained(save_path)
    head_states = {
        'NERmodel_state': model.NERmodel.state_dict(),
        'h2tag_s_state':model.hidden2tag_s.state_dict(),
        'h2tag_t_state':model.hidden2tag_t.state_dict(),
        
        'model_config':{
            'tagset_size':model.tagset_size,
            'num_labels': model.num_labels,
            'evo_dim': model.evo_dim,
            'loss_type': model.loss_type,
            'fasettext_dim': model.fasettext_dim,
            'combined_dim': model.combined_dim,
        }
    }
    torch.save(head_states, os.path.join(save_path, "head_states.pth"))

    print(f"✅ PEFT model, heads, and La-DTL-model saved to: {save_path}")

def load_model_la_dtl(model_class, save_path: str, device: torch.device):
    """
    从 save_path 加载 EvoSegmentLaDTL 模型。
    """
    # 1. 初始化模型实例，不应用 LoRA
    model = model_class(apply_lora_in_init=False).to(device)
    # 2. 加载 LoRA adapter
    print(f"Loading LoRA adapter from {save_path}")
    model.evo = PeftModel.from_pretrained(model.evo, save_path, is_trainable=False)
    # 3. 加载头部状态
    head_path = os.path.join(save_path, 'head_states.pth')
    print(f"Loading head states from {head_path}")
    head_states = torch.load(head_path, map_location=device)

    model.NERmodel.load_state_dict(head_states['NERmodel_state'])
    model.hidden2tag_s.load_state_dict(head_states['h2tag_s_state'])
    model.hidden2tag_t.load_state_dict(head_states['h2tag_t_state'])
  

    # 4. 切换到评估模式
    model.eval()
    print(f"✅ Loaded EvoSegmentLaDTL from {save_path}")
    return model
def save_model_lexicon(model: nn.Module, save_path: str):


    """
    完整保存 LoRA 微调后的 Evo 模型、自定义分类头和 NERmodel。
    - model.evo: 经过 `get_peft_model(...)` 包裹的 PeftModel。
    - model.classifier / model.linear / model.NERmodel: 自定义头部和 NER 模型。
    """
    # 创建保存目录
    os.makedirs(save_path, exist_ok=True)

    # 1. 保存 LoRA adapter
    model.evo.save_pretrained(save_path)

    # 2. 保存分类头和 NERmodel 的状态以及配置信息
    head_states = {
        'classifier_state': model.classifier.state_dict(),
        'linear_state': model.linear.state_dict(),
        'NERmodel_state': model.NERmodel.state_dict(),
        'gated_fusion_state': model.gated_fusion.state_dict(),
        'model_config': {
            'num_labels': model.num_labels,
            'evo_dim': model.evo_dim,
            'loss_type': model.loss_type,
            'kmer_input_dim': getattr(model, 'kmer_input_dim', None),
            'fasettext_dim': getattr(model, 'fasettext_dim', None),
            'combined_dim': getattr(model, 'combined_dim', None),
        }
    }
    torch.save(head_states, os.path.join(save_path, "head_states.pth"))

    print(f"✅ PEFT model, heads, and NERmodel saved to: {save_path}")

def load_model_lexicon(model_class, save_path: str, device: torch.device):
    """
    从 save_path 加载 LoRA 微调后的 Evo 模型、自定义分类头和 NERmodel。
    - model_class: HyenaSegment 类
    - save_path: 保存模型的目录
    - device: torch.device("cuda") 或 CPU
    """
    # 1. 初始化模型（跳过 LoRA 初始化）
    model = model_class(apply_lora_in_init=False).to(device)

    # 2. 加载 LoRA adapter
    print(f"Loading PEFT adapter from: {save_path} onto base model.")
    model.evo = PeftModel.from_pretrained(model.evo, save_path, is_trainable=False)
    print("PEFT adapter loaded.")

    # 3. 加载分类头和 NERmodel 的状态
    head_states_path = os.path.join(save_path, "head_states.pth")
    print(f"Loading head states from: {head_states_path}")
    head_states = torch.load(head_states_path, map_location=device)

    model.classifier.load_state_dict(head_states['classifier_state'])
    model.linear.load_state_dict(head_states['linear_state'])
    model.NERmodel.load_state_dict(head_states['NERmodel_state'])
    if 'gated_fusion_state' in head_states:
        model.gated_fusion.load_state_dict(head_states['gated_fusion_state'])

    # 4. 设置评估模式
    model.eval()
    print(f"✅ PEFT model, heads, and NERmodel loaded from: {save_path}")
    return model
def save_model_complete(model: nn.Module, save_path: str):
    """
    完整保存 LoRA 微调后的 Evo 模型，以及自定义的分类头。
    - model.bert: 已经被 `get_peft_model(...)` 包裹过的 PeftModel。
    - model.classifier / model.linear: 你在 HyenaSegment 里定义的头部。
    """
    import os
    os.makedirs(save_path, exist_ok=True)

    # 1. 保存 LoRA adapter（包括 adapter_config.json、权重、base_model_name_or_path 等）
    #    直接调用 PeftModel.save_pretrained.
    #    注意：HyenaSegment 中用 self.bert = get_peft_model(...) 得到的 self.bert 本身就是 PeftModel 类型，
    #    因此可以直接：
    model.evo.save_pretrained(save_path)

    # 2. 保存分类头的权重到一个单独文件
    head_states = {
        'classifier_state': model.classifier.state_dict(),
        'linear_state': model.linear.state_dict(),
        'model_config': {
            'num_labels': model.num_labels,
            'evo_dim': model.evo_dim,
            'loss_type': getattr(model, 'loss_type', 'focal'),
        }
    }
    torch.save(head_states, os.path.join(save_path, "head_states.pth"))

    print(f"✅ PEFT model and heads saved to: {save_path}")


def load_model_complete(model_class, save_path: str, device: torch.device):
    """
    从 save_path 目录加载 LoRA 微调后的 Evo 模型，以及自定义的分类头。
    - model_class: 你的 HyenaSegment 类
    - save_path: 上面保存时指定的同一目录
    - device: torch.device("cuda") 或 cpu
    """
    from peft import PeftModel, PeftConfig

    # 1. 用同样的 model_class() 构造一个新的实例
    model = model_class(apply_lora_in_init=False).to(device)
    print(f"Loading PEFT adapter from: {save_path} onto base model.")
    model.evo = PeftModel.from_pretrained(model.evo, save_path, is_trainable=False)
    print("PEFT adapter loaded.")
    head_states_path = os.path.join(save_path, "head_states.pth")
    print(f"Loading head states from: {head_states_path}")
    head_states = torch.load(head_states_path, map_location=device)
    
    model.classifier.load_state_dict(head_states['classifier_state'])
    model.linear.load_state_dict(head_states['linear_state']) # 确保linear层确实被使用和保存
    model.eval() # 设置为评估模式
    print(f"✅ PEFT model and heads loaded from: {save_path}")
    return model


# ======================================================================
# EvoSNR 模型存储/读取（新增，不修改任何现有函数）
# 支持 GatedFusion + 双分类头（来自 models/model_evosnr.py 的 EvoSNR）
# ======================================================================

def save_model_evosnr(model, save_path: str):
    """
    保存 EvoSNR 模型（LoRA adapter + 自定义头部）。
    支持的模块：
      - model.evo          ← PeftModel (LoRA adapter)
      - model.NERmodel     ← Transformer NER 编码器
      - model.gated_fusion ← 词汇门控融合模块
      - model.hidden2tag_s ← 源域分类头
      - model.hidden2tag_t ← 目标域分类头
    """
    os.makedirs(save_path, exist_ok=True)

    # 1. 保存 LoRA adapter
    model.evo.save_pretrained(save_path)

    # 2. 保存所有自定义头部权重
    head_states = {
        'NERmodel_state':     model.NERmodel.state_dict(),
        'gated_fusion_state': model.gated_fusion.state_dict(),
        'hidden2tag_s_state': model.hidden2tag_s.state_dict(),
        'hidden2tag_t_state': model.hidden2tag_t.state_dict(),
        'model_config': {
            'tagset_size':    model.tagset_size,
            'num_labels':     model.num_labels,
            'evo_dim':        model.evo_dim,
            'kmer_input_dim': getattr(model, 'kmer_input_dim', 300),
            'combined_dim':   model.combined_dim,
            'loss_type':      model.loss_type,
        }
    }
    torch.save(head_states, os.path.join(save_path, "head_states.pth"))
    print(f"✅ EvoSNR model saved to: {save_path}")


def load_model_evosnr(model_class, save_path: str, device: torch.device):
    """
    加载 EvoSNR 模型（LoRA adapter + 自定义头部）。

    Args:
        model_class: EvoSNR 类（来自 models.model_evosnr）
        save_path:   与 save_model_evosnr 一致的保存目录
        device:      目标设备
    Returns:
        model:       处于 eval 模式的 EvoSNR 实例
    """
    from peft import PeftModel

    # 1. 初始化（跳过 LoRA 初始化，避免重复包裹）
    model = model_class(apply_lora_in_init=False).to(device)

    # 2. 加载 LoRA adapter
    print(f"Loading LoRA adapter from: {save_path}")
    model.evo = PeftModel.from_pretrained(model.evo, save_path, is_trainable=False)
    print("LoRA adapter loaded.")

    # 3. 加载头部权重
    head_path = os.path.join(save_path, "head_states.pth")
    print(f"Loading head states from: {head_path}")
    head_states = torch.load(head_path, map_location=device)

    model.NERmodel.load_state_dict(head_states['NERmodel_state'])
    model.gated_fusion.load_state_dict(head_states['gated_fusion_state'])
    model.hidden2tag_s.load_state_dict(head_states['hidden2tag_s_state'])
    model.hidden2tag_t.load_state_dict(head_states['hidden2tag_t_state'])

    model.eval()
    print(f"✅ EvoSNR model loaded from: {save_path}")
    return model