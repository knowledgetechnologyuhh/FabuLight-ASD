import torch
import torch.nn as nn

from model.Classifier import BGRU
from model.Encoder import face_encoder, audio_encoder, body_encoder

from torch.nn import functional as F

class ASD_Model(nn.Module):
    def __init__(self, bodyPose, upperBody):
        super(ASD_Model, self).__init__()
        
        self.faceEncoder  = face_encoder()
        self.audioEncoder  = audio_encoder()
        
        self.bodyPose = bodyPose
        self.upperBody = upperBody
        if self.bodyPose:
            self.bodyEncoder = body_encoder(3, 128, {'layout' : 'coco_upper' if upperBody else 'coco', 'strategy' : 'spatial'}) #pose_encoder([64, 16], range(11) if upperBody else range(17))
            self.GRUFAB = BGRU(128)
        else:
            self.GRUFA = BGRU(128)
        

    def forward_face_frontend(self, x):
        B, T, W, H = x.shape  
        x = x.view(B, 1, T, W, H)
        x = (x / 255 - 0.4161) / 0.2380
        x = self.faceEncoder(x)
        return x

    def forward_audio_frontend(self, x):
        x = x.unsqueeze(1).transpose(2, 3)
        x = self.audioEncoder(x)
        return x
    
    def forward_body_frontend(self, x):
        x = x.transpose(2, 3).transpose(1, 2).unsqueeze(-1)
        x = self.bodyEncoder(x)
        return x
    
    def forward_face_audio_backend(self, x1, x2):  
        x = x1 + x2 
        x = self.GRUFA(x)   
        x = torch.reshape(x, (-1, 128))
        return x
    
    def forward_face_audio_body_backend(self, x1, x2, x3):        
        x = x1 + x2 + x3
        x = self.GRUFAB(x)   
        x = torch.reshape(x, (-1, 128))
        return x

    def forward_face_backend(self,x):
        x = torch.reshape(x, (-1, 128))
        return x

    def forward_audio_backend(self,x):
        x = torch.reshape(x, (-1, 128))
        return x

    def forward_body_backend(self,x):
        x = torch.reshape(x, (-1, 128))
        return x

    def forward(self, faceFeature, audioFeature, bodyFeature):
        faceEmbed = self.forward_face_frontend(faceFeature)
        audioEmbed = self.forward_audio_frontend(audioFeature)
        outsF = self.forward_face_backend(faceEmbed)
        outsA = self.forward_audio_backend(audioEmbed)
        
        if self.bodyPose:
            bodyEmbed = self.forward_body_frontend(bodyFeature)
            outsB = self.forward_body_backend(bodyEmbed)
            outsFAB = self.forward_face_audio_body_backend(faceEmbed, audioEmbed, bodyEmbed)
            return outsFAB, outsF, outsA, outsB
        else:
            outsFA = self.forward_face_audio_backend(faceEmbed, audioEmbed)
            return outsFA, outsF, outsA
