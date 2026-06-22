"""
model_evosnr.py
================
融合模型：将词汇门控融合（来自 model_lex.py）与 LA-MMD 迁移学习（来自 model_pure_tl.py）合并到同一模型中。

关键设计：
  - 保留 GatedFusion：每个位置自适应决定 EVO 特征 vs k-mer 词汇特征的权重
  - 保留双分类头（hidden2tag_s / hidden2tag_t）：源域和目标域各一个线性分类头
  - 保留 compute_la_mmd / compute_param_transfer_loss 等迁移损失
  - compute_loss 接收双批次（source_batch + target_batch）并带入 k-mer 词汇嵌入
"""

import os
os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"

from transformers import AutoConfig
from peft import LoraConfig, get_peft_model
import torch
import torch.nn as nn

from evo import Evo
from evo.scoring import prepare_batch

from models.layers import NERmodel


class CustomEmbedding(nn.Module):
    """替换 Evo 内部 unembed，使其直接返回隐层向量。"""
    def unembed(self, u):
        return u


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# ====================================================================
# GatedFusion（直接复刻自 model_lex.py）
# ====================================================================
class GatedFusion(nn.Module):
    """
    学习式门控融合：让模型自适应地决定每个位置用多少 EVO 信息 vs k-mer 信息。
    与 model_lex.py 中完全相同的结构。
    """
    def __init__(self, evo_dim: int, kmer_dim: int):
        super().__init__()
        self.kmer_proj = nn.Sequential(
            nn.Linear(kmer_dim, evo_dim),
            nn.LayerNorm(evo_dim),
            nn.GELU(),
        )
        # 门控：基于两种嵌入的拼接来计算每个位置的融合权重
        self.gate = nn.Sequential(
            nn.Linear(evo_dim * 2, evo_dim),
            nn.Sigmoid(),
        )
        self.layer_norm = nn.LayerNorm(evo_dim)

    def forward(self, evo_embed: torch.Tensor, kmer_embed: torch.Tensor):
        """
        Args:
            evo_embed:  (B, L, evo_dim)
            kmer_embed: (B, L, kmer_dim)
        Returns:
            fused:      (B, L, evo_dim)
            gate:       (B, L, evo_dim)  门控权重，可用于可视化
        """
        kmer_proj = self.kmer_proj(kmer_embed)           # (B, L, evo_dim)
        gate_input = torch.cat([evo_embed, kmer_proj], dim=-1)  # (B, L, 2*evo_dim)
        gate = self.gate(gate_input)                     # (B, L, evo_dim)
        fused = gate * evo_embed + (1 - gate) * kmer_proj
        fused = self.layer_norm(fused)
        return fused, gate


# ====================================================================
# EvoSNR：词汇融合 + 迁移学习的统一模型
# ====================================================================
class EvoSNR(nn.Module):
    """
    结合了：
      1. GatedFusion 词汇门控融合（来自 model_lex.py / HyenaSegment）
      2. 双分类头 + LA-MMD 迁移损失（来自 model_pure_tl.py / EvoSegmentPureLaDTL）

    前向推理时，可选传入 kmer_embeddings：
      - 若提供 kmer_embeddings，则使用门控融合
      - 若不提供，则直接使用 EVO 嵌入（退化为纯迁移学习模式）
    """

    def __init__(self, apply_lora_in_init: bool = True):
        super(EvoSNR, self).__init__()

        # ---------- EVO backbone ----------
        self.evo_model_loader = Evo("evo-1-8k-base")
        self.evo_model = self.evo_model_loader.model
        self.tokenizer = self.evo_model_loader.tokenizer

        # ---------- 维度配置 ----------
        self.tagset_size = 2
        self.num_labels = 2
        self.evo_dim = 512
        self.kmer_input_dim = 300   # FastText BME：3 × 100
        self.combined_dim = self.evo_dim
        self.hidden_dim = self.evo_dim

        # ---------- 门控词汇融合模块（来自 model_lex.py）----------
        self.gated_fusion = GatedFusion(
            evo_dim=self.evo_dim,
            kmer_dim=self.kmer_input_dim,
        )

        # ---------- NER Transformer 编码器 ----------
        self.NERmodel = NERmodel(
            model_type="transformer",
            input_dim=self.evo_dim,
            hidden_dim=self.evo_dim,
            num_layer=4,
            biflag=True,
        )

        # ---------- 双分类头（来自 model_pure_tl.py）----------
        # hidden2tag_s：源域分类头
        # hidden2tag_t：目标域分类头
        self.hidden2tag_s = nn.Linear(self.evo_dim, self.tagset_size)
        self.hidden2tag_t = nn.Linear(self.evo_dim, self.tagset_size)



        # ---------- 其他 ----------
        self.dropout = nn.Dropout(0.2)
        self.loss_fn = nn.CrossEntropyLoss()
        self.loss_type = "focal"

        # ---------- EVO 配置 ----------
        hf_model_name = "togethercomputer/evo-1-8k-base"
        model_config = AutoConfig.from_pretrained(
            hf_model_name, trust_remote_code=True, revision="1.1_fix"
        )
        self.evo_model.config = model_config
        self.evo_model.prepare_inputs_for_generation = (
            lambda *args, **kwargs: {"x": args[0], **kwargs}
        )

        # ---------- LoRA 微调 ----------
        if apply_lora_in_init:
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
                ] + [f"blocks.{i}.mlp.l{j}" for i in range(32) for j in range(1, 4)],
            )
            self.evo = get_peft_model(self.evo_model, peft_config)
            print("Applied LoRA in __init__.")
            self.evo.print_trainable_parameters()
        else:
            self.evo = self.evo_model
            print("Skipped LoRA in __init__ (expected during model loading).")

    # ------------------------------------------------------------------
    # 核心嵌入提取
    # 基于 model_pure_tl.py EvoEmb，增加 kmer_embeddings 参数走门控融合
    # ------------------------------------------------------------------
    def extract_representations(self, data_X, kmer_embeddings: torch.Tensor = None):
        self.evo.unembed = CustomEmbedding()

        input_ids, seq_lengths = prepare_batch(
            data_X,
            self.tokenizer,
            prepend_bos=False,
            device=device,
        )
        embed, _ = self.evo(input_ids)
        sequence_output = self.dropout(embed.float())

        if kmer_embeddings is not None:
            kmer_embeddings = kmer_embeddings.to(device)
            fused_embeddings, gate_values = self.gated_fusion(sequence_output, kmer_embeddings)
            fused_embeddings = self.dropout(fused_embeddings)
            feature_out_d = self.NERmodel(fused_embeddings)
        else:
            fused_embeddings = sequence_output
            gate_values = None
            feature_out_d = self.NERmodel(sequence_output)

        attention_weights = self.NERmodel.attention_weights
        return {
            "sequence_output": sequence_output,
            "fused_embeddings": fused_embeddings,
            "feature_out_d": feature_out_d,
            "gate_values": gate_values,
            "attention_weights": attention_weights,
        }

    def EvoEmb(self, data_X, kmer_embeddings: torch.Tensor = None, return_attention: bool = False):
        representations = self.extract_representations(data_X, kmer_embeddings)
        feature_out_d = representations["feature_out_d"]
        gate_values = representations["gate_values"]
        attention_weights = representations["attention_weights"]
        if return_attention:
            return feature_out_d, gate_values, attention_weights
        return feature_out_d, gate_values

    # ------------------------------------------------------------------
    # forward：复用 EvoEmb() 保证训练/推理路径完全一致，
    # 使用 hidden2tag_t（目标域分类头），保留 attn_map 返回方式
    # ------------------------------------------------------------------
    def forward(self, input_seqs=None, label_ids=None, kmer_embeddings=None, attention_mask=None):
        # 复用 EvoEmb 保证 .float() 和 dropout 与训练路径一致
        feature_out_d, gate_values, attention_weights = self.EvoEmb(
            input_seqs, kmer_embeddings, return_attention=True
        )

        # 注意力图（与 model_lex.py 完全一致）
        attn = attention_weights[-1]
        attn_all_heads = attn[0]
        attn_map = attn_all_heads.mean(dim=0)

        # 目标域分类头（使用迁移学习训练的 hidden2tag_t）
        logits = self.hidden2tag_t(feature_out_d)

        outputs = (logits,)
        if label_ids is not None:
            if attention_mask is not None:
                active_loss = attention_mask.view(-1) == 1
                active_logits = logits.view(-1, self.num_labels)[active_loss]
                active_labels = label_ids.view(-1)[active_loss]
            else:
                active_logits = logits.reshape(-1, self.num_labels)
                active_labels = label_ids.reshape(-1)

            outputs = [active_logits, active_labels, feature_out_d, gate_values]

        return outputs, attn_map

    # ------------------------------------------------------------------
    # 迁移学习损失计算（完全复刻 model_pure_tl.py）
    # ------------------------------------------------------------------
    def compute_mmd(self, X: torch.Tensor, Y: torch.Tensor, sigma: float = 1.0):
        """最大均值差异（MMD）。"""
        XX = torch.sum(X ** 2, dim=1, keepdim=True)
        YY = torch.sum(Y ** 2, dim=1, keepdim=True)
        XY = torch.mm(X, Y.t())
        dist_XX = XX - 2 * torch.mm(X, X.t()) + XX.t()
        dist_YY = YY - 2 * torch.mm(Y, Y.t()) + YY.t()
        dist_XY = XX - 2 * XY + YY.t()
        K_XX = torch.exp(-dist_XX / (2 * sigma ** 2))
        K_YY = torch.exp(-dist_YY / (2 * sigma ** 2))
        K_XY = torch.exp(-dist_XY / (2 * sigma ** 2))
        return K_XX.mean() + K_YY.mean() - 2 * K_XY.mean()

    def compute_la_mmd(
        self,
        source_lstm_out: torch.Tensor,
        source_tags: torch.Tensor,
        target_lstm_out: torch.Tensor,
        target_tags: torch.Tensor,
    ):
        """标签对齐 MMD（LA-MMD）。"""
        source_lstm_out = source_lstm_out.reshape(-1, self.hidden_dim)
        source_tags = source_tags.reshape(-1)
        target_lstm_out = target_lstm_out.reshape(-1, self.hidden_dim)
        target_tags = target_tags.reshape(-1)

        source_unique_tags = torch.unique(source_tags)
        target_unique_tags = torch.unique(target_tags)
        common_tags = list(
            set(source_unique_tags.tolist()) & set(target_unique_tags.tolist())
        )

        la_mmd = torch.tensor(0.0, device=source_lstm_out.device)
        for tag in common_tags:
            source_mask = source_tags == tag
            target_mask = target_tags == tag
            if source_mask.sum() > 0 and target_mask.sum() > 0:
                source_features = source_lstm_out[source_mask]
                target_features = target_lstm_out[target_mask]
                la_mmd += self.compute_mmd(source_features, target_features)
        return la_mmd

    def compute_param_transfer_loss(self):
        """参数迁移正则化：约束双分类头权重接近。"""
        weight_s = self.hidden2tag_s.weight
        weight_t = self.hidden2tag_t.weight
        return torch.sum((weight_s - weight_t) ** 2)

    def compute_regularization_loss(self):
        """L2 正则化损失。"""
        loss_r = 0.0
        for param in self.hidden2tag_s.parameters():
            loss_r += torch.sum(param ** 2)
        for param in self.hidden2tag_t.parameters():
            loss_r += torch.sum(param ** 2)
        return loss_r

    def compute_loss(
        self,
        source_data: list,
        target_data: list,
        source_kmer_embeddings: torch.Tensor = None,
        target_kmer_embeddings: torch.Tensor = None,
        index: int = 0,
    ):
        """
        联合损失计算（迁移学习 + 词汇融合）。
        与 model_pure_tl.py 的 compute_loss 完全兼容，
        额外接收 source/target 的 k-mer 词汇嵌入。

        Args:
            source_data: list of (sentence, tags) tuples for source domain
            target_data: list of (sentence, tags) tuples for target domain
            source_kmer_embeddings: (B_s, L, kmer_dim) 可选
            target_kmer_embeddings: (B_t, L, kmer_dim) 可选
            index: 当前 batch index（保留，用于调试或日志）

        Returns:
            total_loss: 标量
        """
        source_sentences = [item[0] for item in source_data]
        source_tags = [item[1] for item in source_data]
        target_sentences = [item[0] for item in target_data]
        target_tags = [item[1] for item in target_data]

        source_batch_size = len(source_sentences)

        # ---------- 嵌入提取（合并为一次 EVO forward，避免重复计算）----------
        combined_sentences = source_sentences + target_sentences

        if source_kmer_embeddings is not None and target_kmer_embeddings is not None:
            # 将源域和目标域的 kmer 嵌入 pad 到相同长度后拼接
            src_kmer_len = source_kmer_embeddings.shape[1]
            tgt_kmer_len = target_kmer_embeddings.shape[1]
            max_kmer_len = max(src_kmer_len, tgt_kmer_len)

            if src_kmer_len < max_kmer_len:
                pad = torch.zeros(
                    source_kmer_embeddings.shape[0],
                    max_kmer_len - src_kmer_len,
                    source_kmer_embeddings.shape[2],
                )
                source_kmer_embeddings = torch.cat([source_kmer_embeddings, pad], dim=1)
            if tgt_kmer_len < max_kmer_len:
                pad = torch.zeros(
                    target_kmer_embeddings.shape[0],
                    max_kmer_len - tgt_kmer_len,
                    target_kmer_embeddings.shape[2],
                )
                target_kmer_embeddings = torch.cat([target_kmer_embeddings, pad], dim=1)

            combined_kmer_embeddings = torch.cat(
                [source_kmer_embeddings, target_kmer_embeddings], dim=0
            )
        else:
            combined_kmer_embeddings = None

        combined_embeds, _ = self.EvoEmb(combined_sentences, combined_kmer_embeddings)
        source_embeds = combined_embeds[:source_batch_size]
        target_embeds = combined_embeds[source_batch_size:]

        # ---------- 计算各域 logits ----------
        emissions_s = self.hidden2tag_s(source_embeds)
        emissions_t = self.hidden2tag_t(target_embeds)

        source_tags = torch.tensor(source_tags).to(device)
        target_tags = torch.tensor(target_tags).to(device)

        # ---------- 交叉熵损失 ----------
        loss_s = self.loss_fn(
            emissions_s.view(-1, self.num_labels), source_tags.view(-1)
        )
        loss_t = self.loss_fn(
            emissions_t.view(-1, self.num_labels), target_tags.view(-1)
        )

        # ---------- 迁移正则化损失 ----------
        loss_la_mmd = self.compute_la_mmd(
            source_embeds, source_tags, target_embeds, target_tags
        )
        loss_p = self.compute_param_transfer_loss()
        loss_r = self.compute_regularization_loss()

        alpha, beta, gamma = 0.1, 0.1, 0.0001
        total_loss = (
            loss_s + loss_t
            + alpha * loss_la_mmd
            + beta * loss_p
            + gamma * loss_r
        )
        return total_loss
