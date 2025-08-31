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
    def __init__(self, num_repr_bases_V, hidden_dim=4):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(num_repr_bases_V, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, num_repr_bases_V)
        )

    def forward(self, x):
        return self.net(x)
        

def get_teacher_model(model_path, feature_dim=512):
    model = resnet18()
    
    if model.fc.in_features != feature_dim:
        raise ValueError(f"Expected feature_dim {feature_dim} does not match ResNet18's fc.in_features of {model.fc.in_features}. Please check your model architecture.")

    model.fc = nn.Identity()
    
    state_dict = torch.load(model_path, map_location='cpu')
    
    if all(key.startswith('backbone.') for key in state_dict.keys()):
        print("stripping 'backbone.' from teacher model state_dict keys")
        state_dict = {k.replace('backbone.', ''): v for k, v in state_dict.items()}

    model.load_state_dict(state_dict)

    for param in model.parameters():
        param.requires_grad = False
    
    model.eval()
    
    return model