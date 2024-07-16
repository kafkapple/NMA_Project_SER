import torch
import torch.nn as nn
import torch.optim as optim
import torch.autograd.profiler as profiler

class SimpleModel(nn.Module):
    def __init__(self):
        super(SimpleModel, self).__init__()
        self.fc = nn.Linear(10, 1)
    
    def forward(self, x):
        return self.fc(x)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = SimpleModel().to(device)
criterion = nn.MSELoss()
optimizer = optim.SGD(model.parameters(), lr=0.01)

inputs = torch.randn(64, 10).to(device)
labels = torch.randn(64, 1).to(device)

# 프로파일링 시작
with profiler.profile(use_cuda=True) as prof:
    for epoch in range(1):
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

# 프로파일링 결과 출력
print(prof.key_averages().table(sort_by="cuda_time_total"))

print(torch.rand(5,3, device='cuda'))

