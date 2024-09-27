import torch

# 입력 텐서
x = torch.tensor([0.2, 0.3, 0.4, 0.5], requires_grad=True) # A2S
y = torch.tensor([1.0, 3.0, 2.0, 4.0], requires_grad=True) # Key

if True:
    tmp = torch.softmax(x/0.1, dim=-1)
    z = y*(1-tmp)
    print(tmp)
else:
    _, indices = torch.topk(x, 2, dim=-1)
    z = y[indices]

loss = (z.sum() - 1.0)

# 역전파
loss.backward()

# 그레디언트 출력
print(y-z)
print(x.grad)
