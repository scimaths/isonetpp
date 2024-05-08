import sys
import torch
import torch.nn as nn
import numpy as np

torch.backends.cuda.matmul.allow_tf32 = False
torch.backends.cudnn.allow_tf32 = False
torch.set_default_dtype(torch.float64)

torch.manual_seed(0)
torch.set_printoptions(precision=30, sci_mode=False)
def my_formatter(x):
    return "%.30f" % x
np.set_printoptions(formatter={'float': my_formatter})
dimin = int(sys.argv[2])
dimout = int(sys.argv[3])
batch = int(sys.argv[4])
class Model(nn.Module):
    def __init__(self):
        super(Model, self).__init__()
        self.linear = nn.Linear(dimin, dimout)

    def forward(self, input):
        output = self.linear(input)
        if sys.argv[1] == '1':
            output *= torch.cat([torch.ones(dimin, dimout) for i in range(batch)], dim=0).to(device)
        else:
            output *= torch.cat([torch.ones(dimin, dimout) for i in range(batch)] + [torch.zeros(dimin, dimout)], dim=0).to(device)
        return torch.sum(output, dim=1)

device = 'cpu'
model = Model().to(device)
model.train()

optimizer = torch.optim.Adam(model.parameters(), lr=0.1, weight_decay=0.1)
optimizer.step()

loss = nn.CrossEntropyLoss()
if sys.argv[1] == '1':
    input = torch.ones(2, batch, dimin, requires_grad=True, device=device)
else:
    input = torch.ones(2, batch + 1, dimin, requires_grad=True, device=device)
target = torch.tensor([0, 1], device=device)
output = model(input)
loss = loss(output, target)
loss.backward()
print('weight')
print(model.linear.weight.grad)
print('bias')
print(model.linear.bias.grad)