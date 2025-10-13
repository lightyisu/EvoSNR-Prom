#only use evo+softmax
import os
os.environ["HF_ENDPOINT"] = "https://alpha.hf-mirror.com"
from transformers import AutoModel, AutoTokenizer,AutoModelForCausalLM,AutoConfig
from peft import LoraConfig, TaskType, get_peft_model
from evo import Evo
from evo.scoring import prepare_batch
import torch.nn as nn
from torchcrf import CRF
from torch.nn import CrossEntropyLoss
import torch



class CustomEmbedding(nn.Module):
  def unembed(self, u):
    return u



device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")
class HyenaSegment(nn.Module):
    def __init__(self, apply_lora_in_init=True):
        super(HyenaSegment, self).__init__()

        self.evo_model_loader=Evo('evo-1-8k-base')
       
        self.evo_model= self.evo_model_loader.model
        self.tokenizer=self.evo_model_loader.tokenizer

        self.num_labels = 2
        self.dropout = nn.Dropout(0.2)
        self.evo_dim=512

       #classifier Block
        self.classifier = nn.Linear(self.evo_dim, self.num_labels)
        self.linear = nn.Linear(512, self.num_labels)
        self.loss_type = 'focal'

       # 将模型和线性层转换为 BFloat16
        self.evo_model = self.evo_model.to(dtype=torch.bfloat16)
        self.classifier = self.classifier.to(dtype=torch.bfloat16)
        self.linear = self.linear.to(dtype=torch.bfloat16)

          # 加载模型配置
        hf_model_name = 'togethercomputer/evo-1-8k-base'
        model_config = AutoConfig.from_pretrained(hf_model_name, trust_remote_code=True, revision='1.1_fix')
        self.evo_model.config = model_config

        
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
                self.evo = get_peft_model(self.evo_model, peft_config)
                print("Applied LoRA in __init__.")
                self.evo.print_trainable_parameters() 
        else:
                self.evo = self.evo_model # 如果不应用LoRA，self.bert就是基础模型
                print("Skipped LoRA in __init__ (expected during model loading).")   
       
        
        

    def forward(self, input_seqs=None, label_ids=None, attention_mask=None):
        
        self.evo.unembed=CustomEmbedding()
        input_ids, seq_lengths = prepare_batch(
            input_seqs,
            self.tokenizer,
            prepend_bos=False,
            device=device,
        )
        embed, _ = self.evo(input_ids) # (batch, length, embed dim)
       
        sequence_output = self.dropout(embed)
        logits= self.classifier(sequence_output)
        
        if label_ids is not None:
            if attention_mask is not None:
                active_loss = attention_mask.view(-1) == 1
                active_logits = logits.view(-1, self.num_labels)[active_loss]
                active_labels = label_ids.view(-1)[active_loss]
            else:
                active_logits = logits.reshape(-1, self.num_labels)
                active_labels = label_ids.reshape(-1)
          
            # outputs = (loss,) + outputs
            outputs = [active_logits,active_labels]

        return outputs