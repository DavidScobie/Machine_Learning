import torch


# define a vgg-16 net
class VGGNet(torch.nn.Module):
    def __init__(self, n_in, n_out):
        super(VGGNet, self).__init__()
        self.ns = [64, -64, 128, -128, 256, 256, -256, 512, 512, 512]  # pooling when negative
        self.n_in = n_in 
        self.n_out = n_out

    def forward(self, x):
        n_pre = self.n_in
        layers = []
        for _n in self.ns:
            layers += self.conv2d_block(n_pre, abs(_n), _n<0)
            n_pre = abs(_n)
        layers += [torch.nn.AvgPool2d(1)]
        x = torch.nn.Sequential(*layers)(x).flatten(1,3)
        return torch.nn.Linear(x.shape[1], self.n_out)(x)
        
    def conv2d_block(self, ch_in, ch_out, post_pooling=False):
        block = [torch.nn.Conv2d(ch_in,ch_out,3), torch.nn.ReLU(), torch.nn.BatchNorm2d(ch_out)]
        if post_pooling:
            block += [torch.nn.MaxPool2d(kernel_size=2, stride=2)]
        return block 


model = VGGNet(3,4)
model.train()

criterion = torch.nn.CrossEntropyLoss(reduction='mean')
optimizer = torch.optim.adam(model.parameters(), lr=1e-4)
for t in range(500):
    # Forward pass: Compute predicted y by passing x to the model
    y_pred = model(x)

    # Compute and print loss
    loss = criterion(y_pred, y)
    if t % 100 == 99:
        print(t, loss.item())

    # Zero gradients, perform a backward pass, and update the weights.
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()