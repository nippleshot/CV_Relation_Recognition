import torch

print("CHECKING GPU")

USE_CUDA = torch.cuda.is_available()
print(USE_CUDA)

device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(device)

