import torch
import torch.nn as nn



# Model Architecture
class DeepLob(nn.Module):
    def __init__(self, y_len):
        super().__init__()
        self.y_len = y_len

        self.conv1 =nn.Sequential(
            nn.Conv2d(in_channels=1, out_channels=32, kernel_size=(1,2), stride=(1,2)),
            nn.LeakyReLU(negative_slope=.01),
            nn.BatchNorm2d(32),

            nn.Conv2d(32,32,kernel_size=(4,1),stride=1),
            nn.LeakyReLU(negative_slope=.01),
            nn.BatchNorm2d(32),

            nn.Conv2d(32,32,kernel_size=(4,1), stride=1),
            nn.LeakyReLU(negative_slope=.01),
            nn.BatchNorm2d(32)   )

        self.conv2 = nn.Sequential(
            nn.Conv2d(32,32,(1,2),(1,2)),
            nn.Tanh() ,
            nn.BatchNorm2d(32),

            nn.Conv2d(32,32,(4,1)),
            nn.Tanh(),
            nn.BatchNorm2d(32),

            nn.Conv2d(32,32,(4,1)),
            nn.Tanh(),
            nn.BatchNorm2d(32),
        )

        self.conv3 = nn.Sequential(
            nn.Conv2d(32,32,(1,10)),
            nn.LeakyReLU(.01),
            nn.BatchNorm2d(32),

            nn.Conv2d(32,32,(4,1)),
            nn.LeakyReLU(.01),
            nn.BatchNorm2d(32),

            nn.Conv2d(32,32,(4,1)),
            nn.LeakyReLU(.01),
            nn.BatchNorm2d(32),

        )

        self.inp1 = nn.Sequential(
            nn.Conv2d(32,64,kernel_size=(1,1),padding='same'),
            nn.LeakyReLU(.01),
            nn.BatchNorm2d(64),

            nn.Conv2d(64,64,kernel_size=(3,1),padding='same'),
            nn.LeakyReLU(.01),
            nn.BatchNorm2d(64),
        )

        self.inp2 = nn.Sequential(
            nn.Conv2d(32,64,kernel_size=(1,1),padding='same'),
            nn.LeakyReLU(.01),
            nn.BatchNorm2d(64),

            nn.Conv2d(64,64,kernel_size=(5,1),padding='same'),
            nn.LeakyReLU(.01),
            nn.BatchNorm2d(64),
        )

        self.inp3 = nn.Sequential(
            nn.MaxPool2d((3,1), stride=(1,1), padding=(1,0)),
            nn.Conv2d(32,64,(1,1),padding='same'),
            nn.LeakyReLU(.01),
            nn.BatchNorm2d(64),
        )

        self.lstm = nn.LSTM(input_size=192, hidden_size=64, num_layers=1, batch_first=True)
        self.fc1 =  nn.Linear(64, self.y_len)


    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)

        x_inp1 = self.inp1(x)
        x_inp2 = self.inp2(x)
        x_inp3 = self.inp3(x)

        x = torch.cat( (x_inp1,x_inp2,x_inp3), dim=1) # (bs,192,sl,1)
        x = x.permute(0,3,2,1) # (bs,1,sl,192)
        x = torch.reshape(x, (-1, x.shape[2], x.shape[3])) # (bs,sl,192)

        x,_ = self.lstm(x) # (bs,sl,64)
        x = x[:,-1,:] #(bs,64)
        x = self.fc1(x) #(bs,y_len)
        return x