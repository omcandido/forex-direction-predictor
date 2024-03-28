import torch.nn as nn
import torch

class ConvNetMultiClass(nn.Module):
    def __init__(self, n_candlesticks):
        super(ConvNetMultiClass, self).__init__()
        self.n_candlesticks = n_candlesticks
        
        self.conv1  = nn.Conv1d(4, 64, 3, padding='same')
        self.conv2  = nn.Conv1d(64, 64, 3, padding='same')
        self.conv3  = nn.Conv1d(64, 128, 3, padding='same')
        self.conv4  = nn.Conv1d(128, 128, 3, padding='same')
        
        self.bn1 = nn.BatchNorm1d(64)
        self.bn2 = nn.BatchNorm1d(64)
        self.bn3 = nn.BatchNorm1d(128)
        self.bn4 = nn.BatchNorm1d(128)
         
        self.fc1 = nn.Linear(3072, 1024)
        self.fc2 = nn.Linear(1024, 128)
        self.fc_out = nn.Linear(128, 3)
        
        self.dropout_conv = nn.Dropout(0.2)
        self.dropout_fc = nn.Dropout(0.5)
        self.pool = nn.MaxPool1d(2)
        
    def forward(self, x):
        out = self.conv1(x)
        out = self.bn1(out)
        out = torch.relu(out)
        out = self.dropout_conv(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = torch.relu(out)
        out = self.dropout_conv(out)
        out = self.pool(out)

        out = self.conv3(out)
        out = self.bn3(out)
        out = torch.relu(out)
        out = self.dropout_conv(out)
        # out = self.pool(out)
        
        out = self.conv4(out)
        out = self.bn4(out)
        out = torch.relu(out)
        out = self.dropout_conv(out)
        out = self.pool(out)

        out = torch.flatten(out, 1) # flatten all dimensions except batch

        out = torch.relu(self.fc1(out))
        out = self.dropout_fc(out)
        
        out = torch.relu(self.fc2(out))
        
        out = self.dropout_fc(out)


        out = self.fc_out(out)
        
        return out