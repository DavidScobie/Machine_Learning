import torch
import numpy as np

folder_name = './data/datasets-promise12'

use_cuda = torch.cuda.is_available()
device = torch.device("cuda:0" if use_cuda else "cpu")

class UNet(torch.nn.Module):

    def __init__(self, ch_in=1, ch_out=1, init_n_feat=32):
        super(UNet, self).__init__()

        n_feat = init_n_feat
        self.encoder1 = UNet._block(ch_in, n_feat)
        self.pool1 = torch.nn.MaxPool3d(kernel_size=2, stride=2)
        self.encoder2 = UNet._block(n_feat, n_feat*2)
        self.pool2 = torch.nn.MaxPool3d(kernel_size=2, stride=2)
        self.encoder3 = UNet._block(n_feat*2, n_feat*4)
        self.pool3 = torch.nn.MaxPool3d(kernel_size=2, stride=2)
        self.encoder4 = UNet._block(n_feat*4, n_feat*8)
        self.pool4 = torch.nn.MaxPool3d(kernel_size=2, stride=2)

        self.bottleneck = UNet._block(n_feat*8, n_feat*16)

        self.upconv4 = torch.nn.ConvTranspose3d(n_feat*16, n_feat*8, kernel_size=3, stride=2)
        self.decoder4 = UNet._block((n_feat*8)*2, n_feat*8)
        self.upconv3 = torch.nn.ConvTranspose3d(n_feat*8, n_feat*4, kernel_size=3, stride=2)
        self.decoder3 = UNet._block((n_feat*4)*2, n_feat*4)
        self.upconv2 = torch.nn.ConvTranspose3d(n_feat*4, n_feat*2, kernel_size=3, stride=2)
        self.decoder2 = UNet._block((n_feat*2)*2, n_feat*2)
        self.upconv1 = torch.nn.ConvTranspose3d(n_feat*2, n_feat, kernel_size=3, stride=2)
        self.decoder1 = UNet._block(n_feat*2, n_feat)

        self.conv = torch.nn.Conv3d(in_channels=n_feat, out_channels=ch_out, kernel_size=1)

    def forward(self, x):
        enc1 = self.encoder1(x)
        enc2 = self.encoder2(self.pool1(enc1))
        enc3 = self.encoder3(self.pool2(enc2))
        enc4 = self.encoder4(self.pool3(enc3))

        bottleneck = self.bottleneck(self.pool4(enc4))

        dec4 = self.upconv4(bottleneck)
        dec4 = torch.cat((dec4, enc4), dim=1)
        dec4 = self.decoder4(dec4)
        dec3 = self.upconv3(dec4)
        dec3 = torch.cat((dec3, enc3), dim=1)
        dec3 = self.decoder3(dec3)
        dec2 = self.upconv2(dec3)
        dec2 = torch.cat((dec2, enc2), dim=1)
        dec2 = self.decoder2(dec2)
        dec1 = self.upconv1(dec2)
        dec1 = torch.cat((dec1, enc1), dim=1)
        dec1 = self.decoder1(dec1)
        return torch.sigmoid(self.conv(dec1))

    @staticmethod
    def _block(ch_in, n_feat):
        return torch.nn.Sequential([
                                    torch.nn.Conv3d(in_channels=ch_in, out_channels=n_feat, kernel_size=3, padding=1, bias=False),
                                    torch.nn.BatchNorm3d(num_features=n_feat),
                                    torch.nn.ReLU(inplace=True),
                                    torch.nn.Conv3d(in_channels=n_feat, out_channels=n_feat, kernel_size=3, padding=1, bias=False),
                                    torch.nn.BatchNorm3d(num_features=n_feat),
                                    torch.nn.ReLU(inplace=True)
                                    ])

class DiceLoss(torch.nn.Module):
    def __init__(self):
        super(DiceLoss, self).__init__()
        self.smooth = 1.0

    def forward(self, y_pred, y_true):
        assert y_pred.size() == y_true.size()
        y_pred = y_pred[:, 0].contiguous().view(-1)
        y_true = y_true[:, 0].contiguous().view(-1)
        intersection = (y_pred * y_true).sum()
        dsc = (2. * intersection + self.smooth) / (y_pred.sum() + y_true.sum() + self.smooth)
        return 1. - dsc


def binary_dice(y_true, y_pred):
    eps = 1e-6
    y_true = y_true >= 0.5
    y_pred = y_pred >= 0.5
    numerator = torch.sum(y_true * y_pred) * 2
    denominator = torch.sum(y_true) + torch.sum(y_pred)
    if numerator == 0 or denominator == 0:
        return 0.0
    else:
        return numerator * 1.0 / denominator


# now define a data loader
class DatasetTrain(torch.utils.data.Dataset):
    def __init__(self, folder_name):
        self.folder_name = folder_name
    
    def __len__(self):
        return self.num_subjects = 50

    def __getitem__(self, index):
        image = np.float32(np.load(os.path.join(self.folder_name, "image_train%02d.npy" % index)))
        label = np.float32(np.load(os.path.join(self.folder_name, "label_train%02d.npy" % index)))       
        return (image, label)


# image_test = np.float32(np.load(os.path.join(self.folder_name, "image_test%02d.npy" % 30)))     


# training
model = VGGNet(1,4)

train_set = H5Dataset(filename)
train_loader = torch.utils.data.DataLoader(
    train_set,
    batch_size=8, 
    shuffle=True,
    num_workers=8)
'''
dataiter = iter(train_loader)
frames, labels = dataiter.next()
'''

criterion = torch.nn.CrossEntropyLoss(reduction='mean')
optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)

freq_print = 200
for epoch in range(20):
    for ii, data in enumerate(train_loader, 0):
        
        moving_loss = 0.0
        frames, labels = data

        optimizer.zero_grad()
        outputs = model(frames)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        # Compute and print loss
        moving_loss += loss.item()
        if ii % freq_print == (freq_print-1):    # print every 2000 mini-batches
            print('[Epoch %d, iter %05d] loss: %.3f' % (epoch, ii, moving_loss/freq_print))
            moving_loss = 0.0

print('Training done.')