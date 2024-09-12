import torch

a = torch.zeros(10, 10, 10)
b = torch.ones(10, 1, 10)
for i in range(1, 11):
    b[:,:,i-1] /= i
c = torch.tensor([1,0,3,5,7,8,9,6,4,5]).unsqueeze(-1).repeat(1,10).unsqueeze(-2)
a.scatter_(-2, c, b)
print(c)
print(a)