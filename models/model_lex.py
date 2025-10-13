#with lexicon k-mer embeddings
#written 20250330
import os
os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"
from transformers import AutoModel, AutoTokenizer,AutoModelForCausalLM,AutoConfig
from peft import LoraConfig, TaskType, get_peft_model
import torch.nn as nn
from evo import Evo
from evo.scoring import prepare_batch
from torch.nn import CrossEntropyLoss
import torch
import numpy as np

import seaborn as sns
import matplotlib.pyplot as plt
from models.layers import NERmodel
from torch import nn

class CustomEmbedding(nn.Module):
  def unembed(self, u):
    return u



#evo-1 pretrain model
#选项：
#1.做 FastText 特征投影 降维度 （目前：不降维度）
#2.EVO 嵌入池化
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
class HyenaSegment(nn.Module):
    def __init__(self,apply_lora_in_init=True):
        super(HyenaSegment, self).__init__()

        self.evo_model_loader=Evo('evo-1-8k-base')
        self.evo_model= self.evo_model_loader.model
        self.tokenizer=self.evo_model_loader.tokenizer
        self.num_labels = 2
        self.dropout = nn.Dropout(0.2)
        
        #FastText Block 
        self.fasettext_dim=1000
        self.evo_dim=512
        self.combined_dim=self.fasettext_dim+self.evo_dim
        


        self.NERmodel = NERmodel(model_type='transformer', input_dim=812, hidden_dim=self.evo_dim, num_layer=4, biflag=True)

        #classifier Block
        # self.classifier = nn.Linear(self.evo_dim, self.num_labels)
        self.classifier = nn.Sequential(
            nn.Linear(self.evo_dim, 256),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(256, self.num_labels)
        )
        self.linear = nn.Linear(512, self.num_labels)
        self.loss_type = 'focal'
        # # 冻结预训练模型self.bert的所有参数
        # for param in self.bert.parameters():
        #     param.requires_grad = False
       
        # 加载模型配置
        hf_model_name = 'togethercomputer/evo-1-8k-base'
        model_config = AutoConfig.from_pretrained(hf_model_name, trust_remote_code=True, revision='1.1_fix')
        self.evo_model.config = model_config

        # # 准备输入生成函数
        # self.bert.prepare_inputs_for_generation = lambda *args, **kwargs: {"input_ids": args[0], **kwargs}
        self.evo_model.prepare_inputs_for_generation = lambda *args, **kwargs: {"x": args[0], **kwargs}
       
        if apply_lora_in_init:
            #Lora finetuning approach
            peft_config = LoraConfig(
                
                inference_mode=False,
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
           
            # 应用 LoRA 微调
            self.evo = get_peft_model(self.evo_model, peft_config)
            self.evo.print_trainable_parameters()  # 打印可训练参数，确认 LoRA 配置生效
        else:
                self.evo = self.evo_model # 如果不应用LoRA，self.bert就是基础模型
                print("Skipped LoRA in __init__ (expected during model loading).")   

    def forward(self, input_seqs=None, label_ids=None,kmer_embeddings=None,attention_mask=None):
        self.evo.unembed=CustomEmbedding()
        kmer_embeddings = kmer_embeddings.to(device)

        input_ids, seq_lengths = prepare_batch(
            input_seqs,
            self.tokenizer,
            prepend_bos=False,
            device=device,
        )
        
        embed, _ = self.evo(input_ids) # (batch, length, embed dim)
        sequence_output = self.dropout(embed)
        combined_embeddings = torch.cat((sequence_output, kmer_embeddings), dim=-1).to(device)
        #(4,180,812)
        combined_embeddings = self.dropout(combined_embeddings)
        feature_out_d = self.NERmodel(combined_embeddings)

        #attention visual
        attn = self.NERmodel.attention_weights[-1]  
        attn_all_heads = attn[0]  
        attn_map = attn_all_heads.mean(dim=0)



        logits= self.classifier(feature_out_d)
        # logits = logits[:, :-1, :]  # 切片操作
        outputs = (logits,) + (combined_embeddings,)
        if label_ids is not None:
            if attention_mask is not None:
                active_loss = attention_mask.view(-1) == 1
                active_logits = logits.view(-1, self.num_labels)[active_loss]
                active_labels = label_ids.view(-1)[active_loss]
            else:
                active_logits = logits.reshape(-1, self.num_labels)
                active_labels = label_ids.reshape(-1)
          
            # outputs = (loss,) + outputs
            outputs = [active_logits,active_labels,feature_out_d]

        return outputs,attn_map
    
