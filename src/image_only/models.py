import timm
import torch.nn as nn
from torchvision import models


class Image_Classifier(nn.Module):
    def __init__(self,model_args):
        super().__init__()
        self.model_args = model_args
        self.model = timm.create_model(model_name=model_args.model_name,pretrained=model_args.pretrained)
        n_features = self.model.head.in_features
        self.model.head = nn.Linear(n_features, 1)
        
    def forward(self, x):
        logits = self.model(x)
        return logits
    
    

class Resnet(nn.Module):
    def __init__(self,model_args):
        super().__init__()
        self.model = models.resnet50(pretrained=model_args.pretrained)
        n_features = self.model.fc.in_features
        self.model.fc = nn.Linear(n_features, 1)
        
    def forward(self, x):
        logits = self.model(x)
        return logits
    
class Resnet101(nn.Module):
    def __init__(self,model_args):
        super().__init__()
        self.model = models.resnet101(pretrained=model_args.pretrained)
        n_features = self.model.fc.in_features
        self.model.fc = nn.Linear(n_features, 1)
        
    def forward(self, x):
        logits = self.model(x)
        return logits
    
class VGG(nn.Module):
    def __init__(self,model_args):
        super().__init__()
        self.model = models.vgg16_bn(pretrained=model_args.pretrained)
        n_features = self.model.classifier[6].in_features
        self.model.classifier[6] = nn.Linear(n_features, 1)
        
    def forward(self, x):
        logits = self.model(x)
        return logits
