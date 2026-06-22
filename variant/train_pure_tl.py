from tqdm import tqdm
import numpy as np
import gc
from torch.utils.data import Dataset, DataLoader, Sampler
from utils.compute_metrics import compute_metrics
from utils.seed import set_seed
from utils.early_stop import EarlyStopping
from utils.bash_config import parse_args
from torch.cuda.amp import autocast
import torch
import torch.optim as optim
import transformers.optimization as op

import Datasets.DataReader_EVO
#need model_pure_tl.py 
import models.model_pure_tl


exec(open("configurator.py").read())
args = parse_args()


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

MaxEpoch = 60
BatchSize = 2


class DualRepeatBalancedBatchSampler(Sampler):
    def __init__(self, source_dataset, target_dataset, batch_size):
        self.source_dataset = source_dataset
        self.target_dataset = target_dataset
        self.batch_size = batch_size

        if batch_size % 2 != 0:
            raise ValueError("BatchSize must be even to ensure balanced source/target batches")

        self.source_len = len(source_dataset)
        self.target_len = len(target_dataset)
        self.num_batches = int(np.ceil(self.source_len / (batch_size // 2)))
        self.target_repeats = max(
            1,
            int(np.ceil((self.num_batches * (batch_size // 2)) / self.target_len)),
        )

    def __iter__(self):
        source_indices = torch.randperm(self.source_len).tolist()

        target_indices = []
        for _ in range(self.target_repeats):
            target_indices.extend(torch.randperm(self.target_len).tolist())
        target_indices = target_indices[:self.num_batches * (self.batch_size // 2)]

        for i in range(self.num_batches):
            s_start = i * (self.batch_size // 2)
            s_end = s_start + (self.batch_size // 2)
            s_batch = source_indices[s_start:s_end]
            if len(s_batch) < self.batch_size // 2:
                needed = self.batch_size // 2 - len(s_batch)
                s_batch += source_indices[:needed]

            t_start = i * (self.batch_size // 2)
            t_end = t_start + (self.batch_size // 2)
            t_batch = target_indices[t_start:t_end]

            yield s_batch, t_batch

    def __len__(self):
        return self.num_batches


class SingleDomainBatchSampler(Sampler):
    def __init__(self, dataset, batch_size):
        self.dataset = dataset
        self.batch_size = batch_size
        self.dataset_len = len(dataset)
        self.num_batches = int(np.ceil(self.dataset_len / batch_size))

    def __iter__(self):
        indices = torch.randperm(self.dataset_len).tolist()
        for i in range(self.num_batches):
            start = i * self.batch_size
            end = start + self.batch_size
            batch = indices[start:end]
            if len(batch) < self.batch_size:
                needed = self.batch_size - len(batch)
                batch += indices[:needed]
            yield batch

    def __len__(self):
        return self.num_batches


class NERDataset(Dataset):
    def __init__(self, sentences, tags):
        self.sentences = sentences
        self.tags = tags

    def __len__(self):
        return len(self.sentences)

    def __getitem__(self, idx):
        return self.sentences[idx], self.tags[idx]


source_path = Source_PCA_path
dev_path = PCA_split_path + "split_dev.csv"
train_path = PCA_split_path + "split_train.csv"
test_path = PCA_split_path + "split_test.csv"

print(f"Loading source: {source_path}")
print(f"Loading target train: {train_path}")

SourceSequenceRaw, SourceLabelRaw, TargetTrainSequenceRaw, TargetTrainLabelRaw = Datasets.DataReader_EVO.DataReaderBERT(
    path1=source_path, path2=train_path
)
SourceSequence, SourceLabel, TargetTrainSequence, TargetTrainLabel = (
    np.array(SourceSequenceRaw),
    np.array(SourceLabelRaw),
    np.array(TargetTrainSequenceRaw),
    np.array(TargetTrainLabelRaw),
)

TargetDevSequenceRaw, TargetDevLabelRaw, TargetTestSequenceRaw, TargetTestLabelRaw = Datasets.DataReader_EVO.DataReaderBERT(
    path1=dev_path, path2=test_path
)

TargetDevSequence, TargetDevLabel, TargetTestSequence, TargetTestLabel = (
    np.array(TargetDevSequenceRaw),
    np.array(TargetDevLabelRaw),
    np.array(TargetTestSequenceRaw),
    np.array(TargetTestLabelRaw),
)

TargetDevLabelRaw = torch.tensor(TargetDevLabelRaw).to(device)
mode = "train"



def predict_domain_logits(neural_network, sentences, domain):
    domain_head = {
        "source": neural_network.hidden2tag_s,
        "target": neural_network.hidden2tag_t,
    }[domain]
    return domain_head(neural_network.EvoEmb(sentences))


def compute_single_domain_loss(neural_network, batch_indices, dataset, domain):
    batch = [dataset[idx] for idx in batch_indices]
    sentences = [item[0] for item in batch]
    tags = torch.tensor([item[1] for item in batch]).to(device)
    logits = predict_domain_logits(neural_network, sentences, domain)
    return neural_network.loss_fn(logits.view(-1, neural_network.num_labels), tags.view(-1))


def compute_train_loss(neural_network, train_batch, transfer_mode, source_dataset, target_dataset, batch_index):
    if transfer_mode == "mixed":
        s_indices, t_indices = train_batch
        source_batch = [source_dataset[idx] for idx in s_indices]
        target_batch = [target_dataset[idx] for idx in t_indices]
        return neural_network.compute_loss(source_batch, target_batch, index=batch_index)

    if transfer_mode == "source_only":
        return compute_single_domain_loss(neural_network, train_batch, source_dataset, "source")

    if transfer_mode == "target_only":
        return compute_single_domain_loss(neural_network, train_batch, target_dataset, "target")


def main():
    if mode == "train":
        print("Enter the Train Mode--------------------->")
        seed = getattr(args, "seed", 1)
        transfer_mode = args.transfer_mode
        set_seed(seed)
        neural_network = models.model_pure_tl.EvoSegmentPureLaDTL().to(device)

        source_dataset = NERDataset(SourceSequenceRaw, SourceLabelRaw)
        target_dataset = NERDataset(TargetTrainSequenceRaw, TargetTrainLabelRaw)
        target_val_dataset = NERDataset(TargetDevSequenceRaw, TargetDevLabelRaw)

        mixed_sampler = DualRepeatBalancedBatchSampler(source_dataset, target_dataset, BatchSize)
        source_sampler = SingleDomainBatchSampler(source_dataset, BatchSize)
        target_sampler = SingleDomainBatchSampler(target_dataset, BatchSize)
        train_samplers = {
            "mixed": mixed_sampler,
            "source_only": source_sampler,
            "target_only": target_sampler,
        }
        train_sampler = train_samplers[transfer_mode]
        val_loader = DataLoader(target_val_dataset, batch_size=BatchSize, shuffle=True)

        t_total = len(train_sampler) * MaxEpoch
        evo_params = list(map(id, neural_network.evo.parameters()))
        downstream_params = filter(lambda p: id(p) not in evo_params, neural_network.parameters())

        optimizer_grouped_parameters = [
            {"params": neural_network.evo.parameters(), "lr": 5e-5},
            {"params": downstream_params, "lr": 2e-4},
        ]

        optimizer = optim.AdamW(optimizer_grouped_parameters, eps=1e-8, betas=(0.9, 0.999))
        scheduler = op.get_linear_schedule_with_warmup(
            optimizer,
            num_warmup_steps=int(t_total * 0.1),
            num_training_steps=t_total,
        )

        print(f"Enter seed mode--------->Now Seed is :{seed}")
        print("Enter the Train Mode--------------------->")
        print(f"Transfer mode: {transfer_mode}")
        print(f"Training sampler: {train_sampler.__class__.__name__}")
        early_stopping = EarlyStopping(patience=15)
        best_mcc = -float("inf")

        for epoch in range(MaxEpoch):
            neural_network.train()
            total_loss = 0.0
            num_batches = len(train_sampler)
            with tqdm(total=num_batches, desc=f"Epoch {epoch + 1}/{MaxEpoch}", unit="batch") as pbar:
                for i, train_batch in enumerate(train_sampler):
                    optimizer.zero_grad()

                    with autocast(enabled=torch.cuda.is_available(), dtype=torch.bfloat16):
                        loss = compute_train_loss(
                            neural_network,
                            train_batch,
                            transfer_mode,
                            source_dataset,
                            target_dataset,
                            batch_index=i,
                        )

                    loss.backward()
                    optimizer.step()
                    scheduler.step()
                    total_loss += loss.item()
                    pbar.set_postfix({"loss": f"{loss.item():.4f}"})
                    pbar.update(1)

            avg_loss = total_loss / num_batches
            print(f"Epoch {epoch + 1}/{MaxEpoch}, Average Loss: {avg_loss:.4f}")

            neural_network.eval()
            val_preds = []
            val_labels = []
            val_probs = []
            total_val_loss = 0.0

            with torch.no_grad():
                for val_batch in tqdm(val_loader, desc=f"Epoch {epoch + 1}/{MaxEpoch} - Validation"):
                    val_sentences, val_tags = val_batch
                    with autocast(enabled=torch.cuda.is_available(), dtype=torch.bfloat16):
                        val_domain = "source" if transfer_mode == "source_only" else "target"
                        logits = predict_domain_logits(neural_network, val_sentences, val_domain)
                        labels_tensor = val_tags.to(device)
                        logits_flat = logits.view(-1, neural_network.num_labels)
                        labels_flat = labels_tensor.view(-1)
                        loss = neural_network.loss_fn(logits_flat, labels_flat)

                    total_val_loss += loss.item()
                    probs = torch.softmax(logits, dim=-1)[:, :, 1]   # 正类概率
                    pred_tags = torch.argmax(logits, dim=-1)
                    pred_tags = pred_tags.cpu().numpy()
                    probs = probs.float().cpu().numpy()
                    for pred_seq, prob_seq, true_seq in zip(pred_tags, probs, val_tags.cpu().numpy()):
                        val_preds.extend(pred_seq)
                        val_probs.extend(prob_seq)
                        val_labels.extend(true_seq[:len(pred_seq)])

            avg_val_loss = total_val_loss / len(val_loader)
            acc, precision, recall, f1, mcc, auprc, jaccard = compute_metrics(
                val_preds, val_labels, positive_probs=val_probs, include_extra=True
            )
            print(f"Validation Results - Epoch {epoch + 1}/{MaxEpoch}:")
            print(f"Average Validation Loss: {avg_val_loss:.4f}")
            print(f"Accuracy: {acc:.4f}")
            print(f"Precision: {precision:.4f}")
            print(f"Recall: {recall:.4f}")
            print(f"F1 Score: {f1:.4f}")
            print(f"MCC: {mcc:.4f}")
            print(f"AUPRC: {auprc:.4f}")
            print(f"Jaccard: {jaccard:.4f}")
            print("-" * 50)
            if mcc > best_mcc:
                best_mcc = mcc
                print(f"New best MCC: {best_mcc:.4f}, saving the model...")

            early_stopping(mcc)
            if early_stopping.early_stop:
                print(f"Early stopping triggered.Best MCC is{best_mcc}")
                neural_network.to("cpu")
                del neural_network
                torch.cuda.empty_cache()
                gc.collect()
                break

    elif mode == "test":
        print("Enter the Test Mode--------------------->")


main()
