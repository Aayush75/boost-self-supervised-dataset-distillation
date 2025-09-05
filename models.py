import os
import torch
import torch.nn as nn
from torchvision.models import resnet18

class ConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=1):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.AvgPool2d(2)
        )
    
    def forward(self, x):
        return self.block(x)
    

class InnerCNN(nn.Module):
    def __init__(self, in_channels=3, channel_list=[128, 256, 512], feature_dim=512):
        super().__init__()
        layers = []
        current_channels = in_channels
        for out_channels in channel_list:
            layers.append(ConvBlock(current_channels, out_channels))
            current_channels = out_channels

        self.features = nn.Sequential(*layers)
        
        self.final_layer = nn.Sequential(
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten(),
            nn.Linear(channel_list[-1], feature_dim)
        )
        
    def forward(self, x):
        x = self.features(x)
        x = self.final_layer(x)
        return x
        
        
class ApproximationMLP(nn.Module):
    def __init__(self, num_repr_bases_V, hidden_dim=256, num_layers=3):
        super().__init__()
        layers = []
        
        if num_layers == 1:
            layers.append(nn.Linear(num_repr_bases_V, num_repr_bases_V))
        else:
            layers.append(nn.Linear(num_repr_bases_V, hidden_dim))
            layers.append(nn.ReLU(inplace=True))
            
            for _ in range(num_layers - 2):
                layers.append(nn.Linear(hidden_dim, hidden_dim))
                layers.append(nn.ReLU(inplace=True))
            
            layers.append(nn.Linear(hidden_dim, num_repr_bases_V))
        
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)
        

def get_teacher_model(model_path, feature_dim=512):
    """Load and initialize teacher model with proper error handling"""
    model = resnet18()
    model.fc = nn.Identity()
    
    if not os.path.exists(model_path):
        print(f"Warning: Teacher model not found at {model_path}")
        print("Using randomly initialized ResNet18 - this will not produce meaningful results!")
        print("Please train a teacher model first using a self-supervised method like SimCLR or Barlow Twins.")
        
        for param in model.parameters():
            param.requires_grad = False
        model.eval()
        return model
    
    try:
        checkpoint = torch.load(model_path, map_location='cpu')
        
        if isinstance(checkpoint, dict):
            if 'state_dict' in checkpoint:
                state_dict = checkpoint['state_dict']
            elif 'model' in checkpoint:
                state_dict = checkpoint['model']
            else:
                state_dict = checkpoint
        else:
            state_dict = checkpoint
            
        if all(key.startswith('backbone.') for key in state_dict.keys()):
            print("Removing 'backbone.' prefix from teacher model state_dict keys")
            state_dict = {k.replace('backbone.', ''): v for k, v in state_dict.items()}
        
        if all(key.startswith('encoder.') for key in state_dict.keys()):
            print("Removing 'encoder.' prefix from teacher model state_dict keys")
            state_dict = {k.replace('encoder.', ''): v for k, v in state_dict.items()}
        
        filtered_state_dict = {}
        for k, v in state_dict.items():
            if k.startswith('fc.') and k != 'fc.weight' and k != 'fc.bias':
                continue
            if 'fc' in k and v.shape[0] != feature_dim:
                continue
            filtered_state_dict[k] = v
        
        model.load_state_dict(filtered_state_dict, strict=False)
        print(f"Successfully loaded teacher model from {model_path}")
        
    except Exception as e:
        print(f"Error loading teacher model from {model_path}: {str(e)}")
        print("Using randomly initialized ResNet18 - this will not produce meaningful results!")

    for param in model.parameters():
        param.requires_grad = False
    
    model.eval()
    return model