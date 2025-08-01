import torch
from LLaMA2 import Transformer, ModelConfig

args = ModelConfig()

x = torch.randint(0, 6144, (1, 50))

model = Transformer(args=args)

num_params = sum(p.numel() for p in model.parameters())
print('Number of paramters:', num_params)

out = model(x)
print(out.logits.shape)