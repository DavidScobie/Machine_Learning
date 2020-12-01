import torch


# define a vgg-16 net
class VGG(torch.nn.Module, ch_in, ch_out):
    def __init__(self):
        super(VGG, self).__init__()
        self.input_layer = conv2d_block(64, ch_in)
        self.output_layer = torch.nn.Linear(512, ch_out)

    def forward(self, x):
        cfg = [64, 64, 128, 128, 256, 256, 256, 512, 512, 512, 512, 512, 512]
        '''
        'VGG11': [64, 128, 256, 256, 512, 512, 512, 512],
        'VGG13': [64, 64, 128, 128, 256, 256, 512, 512, 512, 512],
        'VGG16': [64, 64, 128, 128, 256, 256, 256, 512, 512, 512, 512, 512, 512],
        'VGG19': [64, 64, 128, 128, 256, 256, 256, 256, 512, 512, 512, 512, 512, 512, 512, 512],
        '''
        layers = self.input_layer(x)
        for idx,x in enumerate(cfg):
            layers += [conv2d_block()]
        layers += [torch.nn.AvgPool2d(1)]

        return self.output_layer(torch.nn.Sequential(*layers))

    def conv2d_block(ch_out, ch_in, post_pooling=False):
        block = [torch.nn.Conv2d(ch_in,ch_out,3), torch.nn.ReLU(), torch.nn.BatchNorm2d()]
        if post_pooling:
            block += [torch.nn.MaxPool2d(kernel_size=2, stride=2)]
        return block 




        def test():
    net = VGG('VGG11')
    x = torch.randn(2,3,32,32)
    y = net(x)
    print(y.size())