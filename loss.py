import torch
import torch.nn as nn
import torch.nn.functional as F

class lossFAB(nn.Module):
    def __init__(self):
        super(lossFAB, self).__init__()
        self.criterion = nn.BCELoss()
        self.FC        = nn.Linear(128, 2)
        
    def forward(self, x, labels = None, r = 1):    
        x = x.squeeze(1)
        x = self.FC(x)
        if labels == None:
            predScore = x[:,1]
            predScore = predScore.t()
            predScore = predScore.view(-1).detach().cpu().numpy()
            return predScore
        else:
            x1 = x / r
            x1 = F.softmax(x1, dim = -1)[:,1]
            nloss = self.criterion(x1, labels.float())
            predScore = F.softmax(x, dim = -1)
            predLabel = torch.round(F.softmax(x, dim = -1))[:,1]
            correctNum = (predLabel == labels).sum().float()
            return nloss, x, predScore, predLabel, correctNum

class lossFA(nn.Module):
    def __init__(self):
        super(lossFA, self).__init__()
        self.criterion = nn.BCELoss()
        self.FC        = nn.Linear(128, 2)
        
    def forward(self, x, labels = None, r = 1):    
        x = x.squeeze(1)
        x = self.FC(x)
        if labels == None:
            predScore = x[:,1]
            predScore = predScore.t()
            predScore = predScore.view(-1).detach().cpu().numpy()
            return predScore
        else:
            x1 = x / r
            x1 = F.softmax(x1, dim = -1)[:,1]
            nloss = self.criterion(x1, labels.float())
            predScore = F.softmax(x, dim = -1)
            predLabel = torch.round(F.softmax(x, dim = -1))[:,1]
            correctNum = (predLabel == labels).sum().float()
            return nloss, x, predScore, predLabel, correctNum

class lossF(nn.Module):
    def __init__(self):
        super(lossF, self).__init__()
        self.criterion = nn.BCELoss()
        self.FC        = nn.Linear(128, 2)

    def forward(self, x, labels, r = 1):    
        x = x.squeeze(1)
        x = self.FC(x)
        
        x = x / r
        x = F.softmax(x, dim = -1)

        nloss = self.criterion(x[:,1], labels.float())
        return nloss

class lossB(nn.Module):
    def __init__(self):
        super(lossB, self).__init__()
        self.criterion = nn.BCELoss()
        self.FC        = nn.Linear(128, 2)

    def forward(self, x, labels, r = 1):    
        x = x.squeeze(1)
        x = self.FC(x)
        
        x = x / r
        x = F.softmax(x, dim = -1)

        nloss = self.criterion(x[:,1], labels.float())
        return nloss
