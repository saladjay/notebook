import torch

class AdamGrad(torch.optim.Optimizer):
    def __init__(self, params, lr=1e-3, betas=(0.9, 0.999), eps=1e-8, weight_decay=0):
        defaults = dict(lr=lr, betas=betas, eps=eps, weight_decay=weight_decay)
        super(AdamGrad, self).__init__(params, defaults)
    
    @torch.no_grad()
    def step(self):
        for group in self.param_groups:
            
            lr = group['lr']
            beta1, beta2 = group['betas']
            eps = group['eps']
            weight_decay = group['weight_decay']

            for p in group['params']:
                if p is None:
                    continue
                grad = p.grad

                if grad is None:
                    continue
                
                if weight_decay != 0:
                    # 给grad加上weight_decay的param， 待会会减去这部分的数值，给参数乘于一个alpha是一种正则化的手段，L2正则化
                    grad = grad.add(p, alpha=weight_decay) 
                
                state = self.state[p]
                if len(state) == 0:
                    state['step'] = 0
                    state['exp_avg'] = torch.zeros_like(p, memory_format=torch.preserve_format)
                    state['exp_avg_sq'] = torch.zeros_like(p, memory_format=torch.preserve_format)

                exp_avg = state['exp_avg']
                exp_avg_sq = state['exp_avg_sq']

                state['step'] += 1
                t = state['step']

                # momnentum = beta1 * momnentum + (1 - beta1) * grad
                exp_avg.mul_(beta1).add_(grad, alpha=(1 - beta1))
                # velocity = beta2 * velocity + (1 - beta2) * grad ** 2
                exp_avg_sq.mul_(beta2).addcmul_(grad, grad, value=(1 - beta2))

                bias_correction1 = 1 - beta1 ** t
                bias_correction2 = 1 - beta2 ** t

                exp_avg_hat = exp_avg / bias_correction1
                exp_avg_sq_hat = exp_avg_sq / bias_correction2

                # p = p - lr * momnentum / (sqrt(velocity) + eps)
                denom = exp_avg_sq_hat.sqrt().add_(eps)
                # p = p - lr * exp_avg_hat / denom
                p.addcdiv_(exp_avg_hat, denom, value=-lr)

def test_custom_adam():
    # 初始化相同的参数
    x_custom = torch.tensor([1.0, 2.0], requires_grad=True)
    x_official = torch.tensor([1.0, 2.0], requires_grad=True)
    target = torch.tensor([0.0, 0.0])

    # 使用相同的超参数初始化两个优化器
    custom_optim = AdamGrad([x_custom], lr=0.1, betas=(0.9, 0.999), eps=1e-8)
    official_optim = torch.optim.Adam([x_official], lr=0.1, betas=(0.9, 0.999), eps=1e-8)

    # 进行多轮优化
    for i in range(100):
        # 自定义优化器
        custom_optim.zero_grad()
        loss_custom = (x_custom - target).pow(2).sum()
        loss_custom.backward()
        custom_optim.step()

        # 官方优化器
        official_optim.zero_grad()
        loss_official = (x_official - target).pow(2).sum()
        loss_official.backward()
        official_optim.step()

        # 每隔一定步数打印结果进行比较
        if i % 20 == 0:
            print(f'Step {i}:')
            print(f'  CustomAdam x: {x_custom.detach().numpy()}, loss: {loss_custom.item():.6f}')
            print(f'  Official Adam x: {x_official.detach().numpy()}, loss: {loss_official.item():.6f}')
            print()

    # 最终检查它们是否收敛到非常接近的值
    assert torch.allclose(x_custom, x_official, atol=1e-8), "自定义优化器与官方实现结果差异较大！"
    print("测试通过！自定义 Adam 实现与官方实现行为基本一致。")

test_custom_adam()