import os

from tqdm import tqdm

import numpy as np
import sklearn.model_selection
import torch
import torch.nn as nn
import torch.optim as optim

import Dataset.DataLoader
import Dataset.DataReader
import Model.DeepSNR
import Model.D_AEDNet

import Utils.Metrics
import Utils.Threshold

def set_seed(args):
    np.random.seed(args)
    torch.manual_seed(args)
'''
    Initialization as follows:
'''

NeuralNetworkNameList = ['DeepSNR', 'D_AEDNet']

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
torch.cuda.set_device(0)
use_gpu = torch.cuda.is_available()
print(use_gpu)
'''
    Initialization End    
'''

'''
    Main Process as follows: 
'''
for NeuralNetworkName in NeuralNetworkNameList:
    print(f'---------------------{NeuralNetworkName}------------area----')
   # List to store results for each fold
    fold_results = []

    for seed in range(1, 6):
        set_seed(seed)
    
        FeatureMatrix_train, DenseLabel_train = Dataset.DataReader.DataReader_prom(data_file_path='/root/autodl-tmp/evosnr_0605/data/Agro/split/split_train.csv')
        FeatureMatrix_train, DenseLabel_train = np.array(FeatureMatrix_train), np.array(DenseLabel_train)
        FeatureMatrix_test, DenseLabel_test = Dataset.DataReader.DataReader_prom(data_file_path='/root/autodl-tmp/evosnr_0605/data/Agro/split/split_dev.csv')
        FeatureMatrix_test, DenseLabel_test = np.array(FeatureMatrix_test), np.array(DenseLabel_test)

        ThresholdValue = 0.5

        LossFunction = nn.BCELoss().cuda()

        MaxEpoch = 280

        if NeuralNetworkName == 'DeepSNR':
            MaxEpoch = 80
            NeuralNetwork = Model.DeepSNR.DeepSNR(SequenceLength=180, MotifLength=80).cuda()
        else:
            MaxEpoch = 8000
            NeuralNetwork = Model.D_AEDNet.D_AEDNN(SequenceLength=180).cuda()
        optimizer = optim.Adam(NeuralNetwork.parameters())

        TrainFeatureMatrix = torch.tensor(FeatureMatrix_train, dtype=torch.float32).unsqueeze(dim=1)
        TrainDenseLabels = torch.tensor(DenseLabel_train)
        TrainLoader = Dataset.DataLoader.SampleLoader(FeatureMatrix=TrainFeatureMatrix, DenseLabel=TrainDenseLabels,
                                                        BatchSize=32)
        TestFeatureMatrix = torch.tensor(FeatureMatrix_test, dtype=torch.float32).unsqueeze(dim=1)
        TestDenseLabels = torch.tensor(DenseLabel_test)
        TestLoader = Dataset.DataLoader.SampleLoader(FeatureMatrix=TestFeatureMatrix, DenseLabel=TestDenseLabels,
                                                        BatchSize=32)

        best_mcc = -1
        best_metrics = None
        for Epoch in range(MaxEpoch):
            # train
            NeuralNetwork.train()
            TrainProgressBar = tqdm(TrainLoader)
            for data in TrainProgressBar:
                TrainProgressBar.set_description("Epoch %d" % Epoch)
                optimizer.zero_grad()
                X, Y = data
                X = X.cuda()
                Y = Y.cuda()
                Prediction = NeuralNetwork(X)
                Loss = LossFunction(Prediction.squeeze(), Y.squeeze().to(torch.float32))
                Loss.backward()
                optimizer.step()
            # dev
            NeuralNetwork.eval()
            ValidProgressBar = tqdm(TestLoader)
            pred = np.array([])
            label = np.array([])
            logits = np.array([])
            for data in ValidProgressBar:
                X, Y = data
                X = X.cuda()
                Y = Y.cuda()
                Logits = NeuralNetwork(X)
                Prediction = Utils.Threshold.Threshold(YPredicted=Logits.cpu(), ThresholdValue=ThresholdValue)
                logits = np.append(logits, Logits.cpu().detach().numpy())
                pred = np.append(pred, Prediction)
                label = np.append(label, Y.cpu())
            Performance = np.zeros(shape=7, dtype=np.float32)

            Performance[0], Performance[1], Performance[2], Performance[3], Performance[4], Performance[5],Performance[6] \
                = Utils.Metrics.EvaluationMetricsSequence(y_pred=pred, y_true=label, y_logits=logits)
  
            # Update best MCC and metrics
            if Performance[6] > best_mcc:
                best_mcc = Performance[6]
                best_metrics = Performance.copy()
                torch.save(NeuralNetwork.state_dict(), '/root/autodl-tmp/evosnr_0605/Weights/' + NeuralNetworkName + '/' + 'Agro'+ f'_seed{seed}.pth')
                print('saved with Kleb'+f' _seed{seed}')
          
            acc = Performance[0]
            pre = Performance[1]
            rec = Performance[2]
            f1 = Performance[3]
            auc = Performance[4]
            aupr = Performance[5]
            mcc=Performance[6]
            print(f'----{NeuralNetworkName}----')
     
            print('Acc=%.3f, Pre=%.3f, Rec=%.3f, F1-S=%.3f, AUC=%.3f, AUPR=%.3f, MCC=%.3f' % (
                Performance[0], Performance[1], Performance[2], Performance[3], Performance[4], Performance[5], Performance[6]))
        fold_results.append((seed, best_metrics))
    
            
# # Write results to CSV file
#     csv_file_path = f'/mnt/sdb/data/zwk/Evosnr-main/Experiment/results/results_{NeuralNetworkName}.csv'
#     with open(csv_file_path, 'w') as f:
#         f.write('Fold,Accuracy,Precision,Recall,F1,MCC\n')
#         for seed, metrics in fold_results:
#             f.write(f'{seed},{metrics[0]:.4f},{metrics[1]:.4f},{metrics[2]:.4f},{metrics[3]:.4f},{metrics[6]:.4f}\n')



