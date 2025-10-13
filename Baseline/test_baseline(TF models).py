import os
import numpy as np
from tqdm import tqdm

import torch
import torch.nn as nn

import Dataset.DataLoader
import Dataset.DataReader
import Model.DeepSNR
import Model.D_AEDNet
import Utils.Metrics
import Utils.Threshold

def set_seed(seed):
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
torch.cuda.set_device(0)

NeuralNetworkNameList = ['DeepSNR', 'D_AEDNet']
ThresholdValue = 0.5

for NeuralNetworkName in NeuralNetworkNameList:
    print(f'---------- Test on Testset: {NeuralNetworkName} ----------')
    fold_results = []
    for seed in range(1, 6):
        set_seed(seed)

        # Load test data
        FeatureMatrix_test, DenseLabel_test = Dataset.DataReader.DataReader_prom(
            data_file_path='/root/autodl-tmp/evosnr_0605/data/Agro/split/split_test.csv'
        )
        FeatureMatrix_test = torch.tensor(np.array(FeatureMatrix_test), dtype=torch.float32).unsqueeze(dim=1)
        DenseLabel_test = torch.tensor(np.array(DenseLabel_test))
        TestLoader = Dataset.DataLoader.SampleLoader(FeatureMatrix=FeatureMatrix_test,
                                                     DenseLabel=DenseLabel_test,
                                                     BatchSize=32)

        # Initialize model
        if NeuralNetworkName == 'DeepSNR':
            model = Model.DeepSNR.DeepSNR(SequenceLength=180, MotifLength=80).to(device)
        else:
            #model=model.DeepSNR.DeepSNR
            model = Model.D_AEDNet.D_AEDNN(SequenceLength=180).to(device)

        # Load best weights
        species='Agro'
        weight_path = f'/root/autodl-tmp/evosnr_0605/Weights/{NeuralNetworkName}/{species}_seed{seed}.pth'
        if not os.path.exists(weight_path):
            print(f"Weight not found: {weight_path}")
            continue
        model.load_state_dict(torch.load(weight_path, map_location=device))
        model.eval()

        # Test loop
        pred = np.array([])
        label = np.array([])
        logits = np.array([])

        with torch.no_grad():
            for data in tqdm(TestLoader, desc=f"Testing {NeuralNetworkName} Seed {seed}"):
                X, Y = data
                X = X.to(device)
                Y = Y.to(device)
                Logits = model(X)
                Prediction = Utils.Threshold.Threshold(YPredicted=Logits.cpu(), ThresholdValue=ThresholdValue)

                logits = np.append(logits, Logits.cpu().numpy())
                pred = np.append(pred, Prediction)
                label = np.append(label, Y.cpu().numpy())
        Performance = np.zeros(shape=7, dtype=np.float32)
        Performance[0], Performance[1], Performance[2], Performance[3], Performance[4], Performance[5],Performance[6] \
            = Utils.Metrics.EvaluationMetricsSequence(y_pred=pred, y_true=label, y_logits=logits)
        # Evaluate
        acc, pre, rec, f1, auc, aupr, mcc = Utils.Metrics.EvaluationMetricsSequence(
            y_pred=pred, y_true=label, y_logits=logits
        )
        fold_results.append((seed, Performance))
        print(f"[Seed {seed}] Test Performance:")
        print('Acc=%.3f, Pre=%.3f, Rec=%.3f, F1=%.3f, AUC=%.3f, AUPR=%.3f, MCC=%.3f' % (
            acc, pre, rec, f1, auc, aupr, mcc))
# Write results to CSV file
    csv_file_path = f'/root/autodl-tmp/evosnr_0605/Experiment/results/results_{NeuralNetworkName}_{species}.csv'
    with open(csv_file_path, 'w') as f:
        f.write('Fold,Accuracy,Precision,Recall,F1,MCC\n')
        for seed, metrics in fold_results:
            f.write(f'{seed},{metrics[0]:.4f},{metrics[1]:.4f},{metrics[2]:.4f},{metrics[3]:.4f},{metrics[6]:.4f}\n')