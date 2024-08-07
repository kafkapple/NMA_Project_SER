import torch
import torch.nn as nn
# 768 feature dim from wav2vec
#loss func cross entropy includes softmax

class EmotionRecognitionModel_v1(nn.Module):
    def __init__(self, input_size, num_classes, dropout_rate, activation):
        super(EmotionRecognitionModel_v1, self).__init__()
        self.input_size = input_size
        self.conv1 = nn.Conv1d(input_size, 128, kernel_size=3, padding=1)
        self.conv2 = nn.Conv1d(128, 64, kernel_size=3, padding=1)
        self.pool = nn.AdaptiveAvgPool1d(1)
        self.fc1 = nn.Linear(64, 32)
        self.fc2 = nn.Linear(32, num_classes)

        self.dropout = nn.Dropout(dropout_rate)

        if activation =='gelu':
            self.activation = nn.GELU()
        elif activation =='relu':
            self.activation = nn.GELU()
        elif activation =='leaky_relu':
            self.activation = nn.LeakyReLU(negative_slope=0.01)

    def forward(self, x):
        if x.dim() == 3 and x.shape[1] != self.input_size:
            x = x.transpose(1, 2)
        elif x.dim() == 2:
            x = x.unsqueeze(1)
        else:
            raise ValueError(f"Unexpected shape of input: {x.shape}")
        x = self.activation(self.conv1(x)) 
        x = self.activation(self.conv2(x))
        x = self.pool(x).squeeze(2)
        x = self.activation(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)
        return x

class EmotionRecognitionModel_v2(nn.Module):
    def __init__(self, input_size, num_classes, dropout_rate, activation):
        super(EmotionRecognitionModel_v2, self).__init__()
        self.fc1 = nn.Linear(input_size, 256)
        self.bn1 = nn.BatchNorm1d(256)
        self.fc2 = nn.Linear(256, 128)
        self.bn2 = nn.BatchNorm1d(128)
        self.fc3 = nn.Linear(128, 64)
        self.bn3 = nn.BatchNorm1d(64)
        self.fc4 = nn.Linear(64, num_classes)
        self.dropout = nn.Dropout(dropout_rate)
        
        if activation =='gelu':
            self.activation = nn.GELU()
        elif activation =='relu':
            self.activation = nn.GELU()
        elif activation =='leaky_relu':
            self.activation = nn.LeakyReLU(negative_slope=0.01)
            
    def forward(self, x):
        # print(f"Input shape: {x.shape}")
        x = x.squeeze(1)  # Squeeze the input to remove dimensions of size 1
        # print(f"Shape after squeeze: {x.shape}")
        x = self.activation(self.bn1(self.fc1(x)))
        # print(f"After fc1 and bn1: {x.shape}")
        x = self.dropout(x)
        x = self.activation(self.bn2(self.fc2(x)))
        # print(f"After fc2 and bn2: {x.shape}")
        x = self.dropout(x)
        x = self.activation(self.bn3(self.fc3(x)))
        # print(f"After fc3 and bn3: {x.shape}")
        x = self.dropout(x)
        x = self.fc4(x)
        # print(f"Output shape: {x.shape}")
        return x


