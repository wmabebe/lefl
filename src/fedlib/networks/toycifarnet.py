from torch import nn

    
# reimplemented from tensorflow official tutorials
# http://www.tensorflow.org/tutorials/deep_cnn
class ToyCifarNet(nn.Module):
    def __init__(self, init_weights = True):
        super(ToyCifarNet, self).__init__()

        self.conv0 = nn.Conv2d(3, 32, kernel_size = 3)
        self.conv1 = nn.Conv2d(32, 64, kernel_size = 3)
        self.conv2 = nn.Conv2d(64, 64, kernel_size = 3)
        

        self.fc1 = nn.Linear(1024, 64)
        self.fc2 = nn.Linear(64, 10)

        self.maxpool = nn.MaxPool2d(kernel_size = 2, stride = 2)
        self.avgpool = nn.AdaptiveAvgPool2d(1)

        self.relu = nn.ReLU()

        if init_weights:
            self._initialize_weights()

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        out = self.conv0(x)
        out = self.relu(out)
        out = self.maxpool(out)


        out = self.conv1(out)
        out = self.relu(out)
        out = self.maxpool(out)
        
        out = self.conv2(out)
        out = self.relu(out)


        bs = out.shape[0]

        out = out.view(bs, -1)

        out = self.fc1(out)
        out = self.relu(out)
        out = self.fc2(out)

        return out