import torch
import torch.nn as nn
import torch.nn.functional as F

import sys, time, numpy, os, subprocess, pandas, tqdm
from subprocess import PIPE

from loss import lossF, lossB, lossFA, lossFAB
from model.Model import ASD_Model


class ASD(nn.Module):
    def __init__(self, lr = 0.001, lrDecay = 0.95, bodyPose = True, upperBody = True, optim = "Adam", lossScheduling = "epochStep", numWarmupEpochs = 0., initialTemp = 1., tempDecayType = "linear", tempDecayRate = 0., **kwargs):
        super(ASD, self).__init__()
        
        self.bodyPose = bodyPose
        self.lossScheduling = lossScheduling
        self.numWarmupEpochs = numWarmupEpochs
        self.initialTemp = initialTemp
        self.tempDecayType = tempDecayType
        self.tempDecayRate = tempDecayRate
        
        self.model = nn.DataParallel(ASD_Model(bodyPose, upperBody)).cuda()
        
        self.lossF = nn.DataParallel(lossF()).cuda()
        
        if self.bodyPose:
            self.lossFAB= nn.DataParallel(lossFAB()).cuda()
            self.lossB = nn.DataParallel(lossB()).cuda()
        else:
            self.lossFA = nn.DataParallel(lossFA()).cuda()
        
        self.initial_lr = lr
        self.lrDecay = lrDecay
        if optim == "Adam":
            self.optim = torch.optim.Adam(self.parameters(), lr = lr)
        elif optim == "AdamW":
            self.optim = torch.optim.AdamW(self.parameters(), lr = lr)
        else:
            raise Exception(f"{optim} is not a valid optimiser type.")
        
        #self.scheduler = torch.optim.lr_scheduler.StepLR(self.optim, step_size = 1, gamma=lrDecay)
        print(time.strftime("%m-%d %H:%M:%S") + " Model para number = %.2f"%(sum(param.numel() for param in self.model.parameters()) / 1000 / 1000))

    def train_network(self, loader, epoch, **kwargs):
        self.train()
        index, top1, lossF, lossB, lossFA, lossFAB, loss = 0, 0, 0, 0, 0, 0, 0
        
        if self.tempDecayType == "linear":
            r = self.initialTemp + self.tempDecayRate * (1 - epoch)
        elif self.tempDecayType == "exp":
            r = self.initialTemp * self.tempDecayRate ** (1 - epoch)
        else:
            raise Exception(f"{self.tempDecayType} is not a valid temperature decay type.")
        
        for num, (faceFeature, audioFeature, bodyFeature, labels) in enumerate(loader, start=1):            
            if (epoch - 1 + num / len(loader)) < self.numWarmupEpochs:
                self.optim.param_groups[0]['lr'] = self.initial_lr * (epoch - 1 + num / len(loader)) / self.numWarmupEpochs
            elif self.lossScheduling == "epochStep":
                self.optim.param_groups[0]['lr'] = self.initial_lr * self.lrDecay ** (epoch - numpy.ceil(self.numWarmupEpochs) - 1)
            elif self.lossScheduling == "batchStep":
                self.optim.param_groups[0]['lr'] = self.initial_lr * self.lrDecay ** (epoch - self.numWarmupEpochs - 1 + num / len(loader))
            else:
                raise Exception(f"{self.lossScheduling} is not a valid loss scheduling scheme.")
            
            lr = self.optim.param_groups[0]['lr']
            self.zero_grad()
            
            labels = labels[0].reshape((-1)).cuda() 
            
            if self.bodyPose:
                outsFAB, outsF, _, outsB = self.model(faceFeature[0].cuda(), audioFeature[0].cuda(), bodyFeature[0].cuda())
                
                nlossFAB, _, _, prec = self.lossFAB.forward(outsFAB, labels, r)
                nlossF = self.lossV.forward(outsF, labels, r)
                nlossB = self.lossb.forward(outsB, labels, r)
                nloss = nlossFAB + 0.25 * (nlossF + nlossB)
                
                lossFAB += nlossFAB.mean().detach().cpu().numpy()
                lossF += nlossF.mean().detach().cpu().numpy()
                lossB += nlossB.mean().detach().cpu().numpy()
            else:
                outsFA, outsF, _ = self.model(faceFeature[0].cuda(), audioFeature[0].cuda(), None)
                
                nlossFA, _, _, prec = self.lossFA.forward(outsFA, labels, r)
                nlossF = self.lossF.forward(outsF, labels, r)
                nloss = nlossFA + 0.5 * nlossF
                
                lossFA += nlossFA.mean().detach().cpu().numpy()
                lossF += nlossF.mean().detach().cpu().numpy()
            
            loss += nloss.mean().detach().cpu().numpy()
            top1 += prec.sum()
            nloss.mean().backward()
            self.optim.step()
            index += len(labels)
            
            if self.bodyPose:
                sys.stderr.write(time.strftime("%m-%d %H:%M:%S") + \
                " [%2d] r: %2f, Lr: %5f, Training: %.2f%%, "    %(epoch, r, lr, 100 * (num / loader.__len__())) + \
                " LossF: %.5f, LossB: %.5f, LossFAB: %.5f, Loss: %.5f, ACC: %2.2f%% \r"  %(lossF/(num), lossB/(num), lossFAB/(num), loss/(num), 100 * (top1/index)))
            else:
                sys.stderr.write(time.strftime("%m-%d %H:%M:%S") + \
                " [%2d] r: %2f, Lr: %5f, Training: %.2f%%, "    %(epoch, r, lr, 100 * (num / loader.__len__())) + \
                " LossF: %.5f, LossFA: %.5f, Loss: %.5f, ACC: %2.2f%% \r"  %(lossF/(num), lossFA/(num), loss/(num), 100 * (top1/index)))
            sys.stderr.flush()
        
        sys.stdout.write("\n")      

        return loss/num, lr

    def evaluate_network(self, loader, evalCsvSave, evalOrig, **kwargs):
        self.eval()
        all_outputs = None
        predScores = []
        for faceFeature, audioFeature, bodyFeature, labels in tqdm.tqdm(loader):
            with torch.no_grad():
                labels = labels[0].reshape((-1)).cuda()
                
                if self.bodyPose:
                    outsFAB, _, _, _= self.model(faceFeature[0].cuda(), audioFeature[0].cuda(), bodyFeature[0].cuda())
                    _, x, predScore, _, _ = self.lossFAB.forward(outsFAB, labels)
                    if all_outputs == None:
                        all_outputs = x
                    else:
                        all_outputs = torch.cat((all_outputs, x), 0)
                else:
                    outsFA, _, _ = self.model(faceFeature[0].cuda(), audioFeature[0].cuda(), None)
                    _, x, predScore, _, _ = self.lossFA.forward(outsFA, labels)
                    if all_outputs == None:
                        all_outputs = x
                    else:
                        all_outputs = torch.cat((all_outputs, x), 0)
                predScore = predScore[:,1].detach().cpu().numpy()
                predScores.extend(predScore)
        
        print(torch.std_mean(all_outputs, dim=0, keepdim=True))
        sys.exit(1)
        
        evalLines = open(evalOrig).read().splitlines()[1:]
        labels = []
        labels = pandas.Series( ['SPEAKING_AUDIBLE' for line in evalLines])
        scores = pandas.Series(predScores)
        evalRes = pandas.read_csv(evalOrig)
        evalRes['score'] = scores
        evalRes['label'] = labels
        evalRes.drop(['label_id'], axis=1,inplace=True)
        evalRes.drop(['instance_id'], axis=1,inplace=True)
        evalRes.to_csv(evalCsvSave, index=False)
        cmd = "python3 -O ./eval/WASD_evaluation.py -g %s -p %s "%(evalOrig, evalCsvSave)
        res = str(subprocess.run(cmd, shell=True, stdout=PIPE, stderr=PIPE).stdout)[2 : -3].split("\\n")
        mapdata = {r.split(' ')[0] : float(r.split(' ')[-1][:-1]) for r in res}
        return mapdata

    def saveParameters(self, path):
        torch.save(self.state_dict(), path)

    def loadParameters(self, path):
        selfState = self.state_dict()
        loadedState = torch.load(path)
        for name, param in loadedState.items():
            origName = name;
            if name not in selfState:
                name = name.replace("module.", "")
                if name not in selfState:
                    print("%s is not in the model."%origName)
                    continue
            if selfState[name].size() != loadedState[origName].size():
                sys.stderr.write("Wrong parameter length: %s, model: %s, loaded: %s"%(origName, selfState[name].size(), loadedState[origName].size()))
                continue
            selfState[name].copy_(param)
