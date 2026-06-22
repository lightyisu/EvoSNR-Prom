import os
os.environ["HF_ENDPOINT"] = "https://alpha.hf-mirror.com"

from transformers import AutoConfig
from peft import LoraConfig, get_peft_model
import torch
import torch.nn as nn

from evo import Evo
from evo.scoring import prepare_batch

from models.layers import NERmodel


class CustomEmbedding(nn.Module):
    def unembed(self, u):
        return u


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class EvoSegmentPureLaDTL(nn.Module):
    def __init__(self, apply_lora_in_init=True):
        super(EvoSegmentPureLaDTL, self).__init__()

        self.tagset_size = 2
        self.evo_model_loader = Evo("evo-1-8k-base")
        self.evo_model = self.evo_model_loader.model
        self.tokenizer = self.evo_model_loader.tokenizer

        self.num_labels = 2
        self.dropout = nn.Dropout(0.2)
        self.evo_dim = 512
        self.combined_dim = self.evo_dim

        self.NERmodel = NERmodel(
            model_type="transformer",
            input_dim=self.combined_dim,
            hidden_dim=self.evo_dim,
            num_layer=4,
            biflag=True,
        )

        self.hidden2tag_s = nn.Linear(self.evo_dim, self.tagset_size)
        self.hidden2tag_t = nn.Linear(self.evo_dim, self.tagset_size)
        self.hidden_dim = self.evo_dim
        self.loss_fn = nn.CrossEntropyLoss()
        self.loss_type = "focal"

        hf_model_name = "togethercomputer/evo-1-8k-base"
        model_config = AutoConfig.from_pretrained(
            hf_model_name, trust_remote_code=True, revision="1.1_fix"
        )
        self.evo_model.config = model_config
        self.evo_model.prepare_inputs_for_generation = lambda *args, **kwargs: {"x": args[0], **kwargs}

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
                ] + [f"blocks.{i}.mlp.l{j}" for i in range(32) for j in range(1, 4)]
            )
            self.evo = get_peft_model(self.evo_model, peft_config)
            print("Applied LoRA in __init__.")
            self.evo.print_trainable_parameters()
        else:
            self.evo = self.evo_model
            print("Skipped LoRA in __init__ (expected during model loading).")

    def forward(self, input_seqs=None, label_ids=None, attention_mask=None, domain="target"):
        self.evo.unembed = CustomEmbedding()

        input_ids, seq_lengths = prepare_batch(
            input_seqs,
            self.tokenizer,
            prepend_bos=False,
            device=device,
        )

        embed, _ = self.evo(input_ids)
        sequence_output = self.dropout(embed.float())
        feature_out_d = self.NERmodel(sequence_output)
        if domain == "source":
            logits = self.hidden2tag_s(feature_out_d)
        else:
            logits = self.hidden2tag_t(feature_out_d)

        if label_ids is not None:
            if attention_mask is not None:
                active_loss = attention_mask.view(-1) == 1
                active_logits = logits.view(-1, self.num_labels)[active_loss]
                active_labels = label_ids.view(-1)[active_loss]
            else:
                active_logits = logits.view(-1, self.num_labels)
                active_labels = label_ids.view(-1)
            return [active_logits, active_labels]
        return logits

    def compute_mmd(self, X, Y, sigma=1.0):
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

    def compute_la_mmd(self, source_lstm_out, source_tags, target_lstm_out, target_tags):
        source_lstm_out = source_lstm_out.reshape(-1, self.hidden_dim)
        source_tags = source_tags.reshape(-1)
        target_lstm_out = target_lstm_out.reshape(-1, self.hidden_dim)
        target_tags = target_tags.reshape(-1)
        source_unique_tags = torch.unique(source_tags)
        target_unique_tags = torch.unique(target_tags)
        common_tags = list(set(source_unique_tags.tolist()) & set(target_unique_tags.tolist()))
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
        weight_s = self.hidden2tag_s.weight
        weight_t = self.hidden2tag_t.weight
        return torch.sum((weight_s - weight_t) ** 2)

    def compute_regularization_loss(self):
        loss_r = 0.0
        for param in self.hidden2tag_s.parameters():
            loss_r += torch.sum(param ** 2)
        for param in self.hidden2tag_t.parameters():
            loss_r += torch.sum(param ** 2)
        return loss_r

    def EvoEmb(self, data_X, return_attention=False):
        self.evo.unembed = CustomEmbedding()

        input_ids, seq_lengths = prepare_batch(
            data_X,
            self.tokenizer,
            prepend_bos=False,
            device=device,
        )
        embed, _ = self.evo(input_ids)
        sequence_output = self.dropout(embed.float())
        feature_out_d = self.NERmodel(sequence_output)
        attention_weights = self.NERmodel.attention_weights
        if return_attention:
            return feature_out_d, attention_weights
        return feature_out_d

    def compute_loss(self, source_data, target_data, index):
        source_sentences = [item[0] for item in source_data]
        source_tags = [item[1] for item in source_data]
        target_sentences = [item[0] for item in target_data]
        target_tags = [item[1] for item in target_data]

        source_batch_size = len(source_sentences)
        combined_sentences = source_sentences + target_sentences
        combined_embeds = self.EvoEmb(combined_sentences)
        source_embeds = combined_embeds[:source_batch_size]
        target_embeds = combined_embeds[source_batch_size:]

        emissions_s = self.hidden2tag_s(source_embeds)
        emissions_t = self.hidden2tag_t(target_embeds)

        source_tags = torch.tensor(source_tags).to(device)
        target_tags = torch.tensor(target_tags).to(device)

        loss_s = self.loss_fn(emissions_s.view(-1, self.num_labels), source_tags.view(-1))
        loss_t = self.loss_fn(emissions_t.view(-1, self.num_labels), target_tags.view(-1))

        loss_la_mmd = self.compute_la_mmd(source_embeds, source_tags, target_embeds, target_tags)
        loss_p = self.compute_param_transfer_loss()
        loss_r = self.compute_regularization_loss()

        alpha, beta, gamma = 0.1, 0.1, 0.0001
        total_loss = loss_s + loss_t + alpha * loss_la_mmd + beta * loss_p + gamma * loss_r
        return total_loss
