import torch
import torch.nn as nn
import snntorch as snn
import torch.nn.functional as F


# 标准ResNet18
class BasicBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, 3, stride, 1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU()
        self.conv2 = nn.Conv2d(out_channels, out_channels, 3, 1, 1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)
        
        self.shortcut = nn.Sequential()
        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, 1, stride, bias=False),
                nn.BatchNorm2d(out_channels)
            )
    
    def forward(self, x):
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        out = self.relu(out)
        return out

class ResNet18(nn.Module):
    def __init__(self, num_classes=10):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 64, 7, 2, 3, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU()
        self.maxpool = nn.MaxPool2d(3, 2, 1)
        
        self.layer1 = self._make_layer(64, 64, 2, 1)
        self.layer2 = self._make_layer(64, 128, 2, 2)
        self.layer3 = self._make_layer(128, 256, 2, 2)
        self.layer4 = self._make_layer(256, 512, 2, 2)
        
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(512, num_classes)
    
    def _make_layer(self, in_channels, out_channels, blocks, stride):
        layers = []
        layers.append(BasicBlock(in_channels, out_channels, stride))
        for _ in range(1, blocks):
            layers.append(BasicBlock(out_channels, out_channels))
        return nn.Sequential(*layers)
    
    def forward(self, x):
        x = self.relu(self.bn1(self.conv1(x)))
        x = self.maxpool(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)
        return x

# 使用LIF神经元的SNN-ResNet18
class SNNBasicBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1, beta=0.9):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, 3, stride, 1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.lif1 = snn.Leaky(beta=beta)
        self.conv2 = nn.Conv2d(out_channels, out_channels, 3, 1, 1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.lif2 = snn.Leaky(beta=beta)
        
        self.shortcut = nn.Sequential()
        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, 1, stride, bias=False),
                nn.BatchNorm2d(out_channels)
            )
    
    def forward(self, x):
        mem1 = self.lif1.init_leaky()
        mem2 = self.lif2.init_leaky()
        
        out = self.bn1(self.conv1(x))
        out, mem1 = self.lif1(out, mem1)
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        out, mem2 = self.lif2(out, mem2)
        return out

class SNNResNet18(nn.Module):
    def __init__(self, num_classes=10, beta=0.9, num_steps=10):
        super().__init__()
        self.num_steps = num_steps
        self.conv1 = nn.Conv2d(3, 64, 7, 2, 3, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.lif1 = snn.Leaky(beta=beta)
        self.maxpool = nn.MaxPool2d(3, 2, 1)
        
        self.layer1 = self._make_layer(64, 64, 2, 1, beta)
        self.layer2 = self._make_layer(64, 128, 2, 2, beta)
        self.layer3 = self._make_layer(128, 256, 2, 2, beta)
        self.layer4 = self._make_layer(256, 512, 2, 2, beta)
        
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(512, num_classes)
        self.lif_out = snn.Leaky(beta=beta, output=True)
    
    def _make_layer(self, in_channels, out_channels, blocks, stride, beta):
        layers = []
        layers.append(SNNBasicBlock(in_channels, out_channels, stride, beta))
        for _ in range(1, blocks):
            layers.append(SNNBasicBlock(out_channels, out_channels, beta=beta))
        return nn.Sequential(*layers)
    
    def forward(self, x):
        mem1 = self.lif1.init_leaky()
        mem_out = self.lif_out.init_leaky()
        
        spk_rec = []
        mem_rec = []
        
        for step in range(self.num_steps):
            cur_x = x
            cur_x = self.bn1(self.conv1(cur_x))
            cur_x, mem1 = self.lif1(cur_x, mem1)
            cur_x = self.maxpool(cur_x)
            
            cur_x = self.layer1(cur_x)
            cur_x = self.layer2(cur_x)
            cur_x = self.layer3(cur_x)
            cur_x = self.layer4(cur_x)
            
            cur_x = self.avgpool(cur_x)
            cur_x = torch.flatten(cur_x, 1)
            cur_x = self.fc(cur_x)
            
            spk, mem_out = self.lif_out(cur_x, mem_out)
            spk_rec.append(spk)
            mem_rec.append(mem_out)
        
        return torch.stack(spk_rec, dim=0), torch.stack(mem_rec, dim=0)




import torch
import torch.nn as nn
from torchvision.models import resnet18, ResNet18_Weights

#为什么修改全连接层，因为输出的脉冲稀疏，无法保证单峰性
#参数量：self.mu: shape = (in_features, 1) = (512, 1) = 512个参数
#self.sigma: shape = (in_features, 1) = (512, 1) = 512个参数
#self.amplitude: shape = (in_features, 1) = (512, 1) = 512个参数
#self.bias: shape = (out_features,) = (100,) = 100个参数
#非可训练参数（buffer）：
#5. self.positions: shape = (1, out_features) = (1, 100) = 100个参数
#可训练参数: 512 + 512 + 512 + 100 = 1,636个


class GaussianFC(nn.Module):
    def __init__(self, in_features, out_features, bias=True):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        
        # 每个输入神经元的高斯分布参数
        self.mu = nn.Parameter(torch.rand(in_features, 1) * out_features)  # 峰值位置
        self.sigma = nn.Parameter(torch.ones(in_features, 1) * 10)  # 分布宽度
        self.amplitude = nn.Parameter(torch.ones(in_features, 1))  # 峰值高度
        
        # 输出位置索引
        self.register_buffer('positions', torch.arange(out_features).float().unsqueeze(0))
        # 添加偏置项
        if bias:
            self.bias = nn.Parameter(torch.zeros(out_features))
        else:
            self.register_parameter('bias', None)
    def forward(self, x):
        # 计算高斯分布形状的权重
        # weight[i,j] = amplitude[i] * exp(-(j - mu[i])^2 / (2 * sigma[i]^2))
        #每个输入数据到输出的权重都是正在分布。但最后的总输出未必是正态分布。
        sigle_feature_Gaussian = True
        if sigle_feature_Gaussian:
            diff = self.positions - self.mu  # (in_features, out_features)
            weight = self.amplitude * torch.exp(-diff**2 / (2 * self.sigma**2))
        #让最后的输出结果是为正，且正态分布。
        output = x @ weight
        if self.bias is not None:
            output = F.relu(output + self.bias)
        return output


#GaussianFC会导致：出现多个单峰性

#1 限制μ的范围，防止多峰重叠

class GaussianFC_uni(nn.Module):
    def __init__(self, in_features, out_features, bias=True):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        
        # 将μ限制在更小的范围内，避免多个高斯分布重叠产生多峰
        self.mu = nn.Parameter(torch.rand(in_features, 1) * (out_features * 0.3) + out_features * 0.35)
        self.sigma = nn.Parameter(torch.ones(in_features, 1) * 5)  # 减小σ，让峰更集中
        self.amplitude = nn.Parameter(torch.ones(in_features, 1))
        
        self.register_buffer('positions', torch.arange(out_features).float().unsqueeze(0))
        if bias:
            self.bias = nn.Parameter(torch.zeros(out_features))
        else:
            self.register_parameter('bias', None)
    def forward(self, x):
        # 可以对μ进行约束，确保它们不会分散太远
        constrained_mu = torch.sigmoid(self.mu) * self.out_features  # 将μ限制在[0, out_features]
        
        diff = self.positions - constrained_mu
        weight = self.amplitude * torch.exp(-diff**2 / (2 * self.sigma**2))

        output = x @ weight
        if self.bias is not None:
            output = F.relu(output + self.bias)   

        return output


# 使用单个全局高斯分布

class SingleGaussianFC(nn.Module):
    """高斯分布的中心位置可以根据输入变化"""
    def __init__(self, in_features, out_features):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features

        # 用于预测高斯参数的线性层
        self.mu_predictor = nn.Linear(in_features, 1)      # 预测中心位置
        self.sigma_predictor = nn.Linear(in_features, 1)   # 预测宽度
        self.amplitude_predictor = nn.Linear(in_features, 1)  # 预测幅度
        
        # 初始化预测器，让输出在合理范围内
        with torch.no_grad():
            # μ初始化到中央区域
            self.mu_predictor.weight.normal_(0, 0.1)
            self.mu_predictor.bias.fill_(out_features / 2)
            
            # σ初始化到合理宽度
            self.sigma_predictor.weight.normal_(0, 0.1)
            self.sigma_predictor.bias.fill_(out_features / 4)
            
            # amplitude初始化
            self.amplitude_predictor.weight.normal_(0, 0.1)
            self.amplitude_predictor.bias.fill_(1.0)
        
        # 位置索引
        self.register_buffer('positions', torch.arange(out_features).float())
    
    def forward(self, x):
        batch_size = x.size(0)
        
        # 根据输入预测每个样本的高斯参数
        mu = self.mu_predictor(x)  # (batch_size, 1)
        sigma = self.sigma_predictor(x)  # (batch_size, 1)
        amplitude = self.amplitude_predictor(x)  # (batch_size, 1)
        
        # 确保参数在合理范围内
        mu = torch.sigmoid(mu) * self.out_features  # 限制在 [0, out_features]
        sigma = F.softplus(sigma)  # 确保 σ > 0
        amplitude = F.softplus(amplitude)  # 确保幅度 > 0
        
        # 计算每个样本的高斯分布
        positions_expanded = self.positions.unsqueeze(0).expand(batch_size, -1)  # (batch_size, out_features)
        mu_expanded = mu.expand(-1, self.out_features)  # (batch_size, out_features)
        sigma_expanded = sigma.expand(-1, self.out_features)  # (batch_size, out_features)
        amplitude_expanded = amplitude.expand(-1, self.out_features)  # (batch_size, out_features)
        
        # 生成高斯分布
        diff = positions_expanded - mu_expanded
        gaussian_output = amplitude_expanded * torch.exp(-diff**2 / (2 * sigma_expanded**2))
        
        return gaussian_output


"""FlexibleSingleGaussianFC可以选择哪些参数是可变的
model2 = FlexibleSingleGaussianFC(in_features, out_features, 
                                     learnable_mu=True, 
                                     learnable_sigma=False, 
                                     learnable_amplitude=False)"""
class FlexibleSingleGaussianFC(nn.Module):

    def __init__(self, in_features, out_features, 
                 learnable_mu=True, learnable_sigma=True, learnable_amplitude=True):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.learnable_mu = learnable_mu 
        self.learnable_sigma = learnable_sigma
        self.learnable_amplitude = learnable_amplitude
        
        # 根据配置创建参数预测器或固定参数
        if learnable_mu: #预测中心位置
            self.mu_predictor = nn.Linear(in_features, 1)
            with torch.no_grad():
                self.mu_predictor.weight.normal_(0, 0.1)
                self.mu_predictor.bias.fill_(out_features / 2)
        else:
            self.register_parameter('fixed_mu', nn.Parameter(torch.tensor(out_features / 2.0)))
        
        if learnable_sigma:# 预测宽度
            self.sigma_predictor = nn.Linear(in_features, 1)
            with torch.no_grad():
                self.sigma_predictor.weight.normal_(0, 0.1)
                self.sigma_predictor.bias.fill_(out_features / 4)
        else:
            self.register_parameter('fixed_sigma', nn.Parameter(torch.tensor(out_features / 4.0)))
        
        if learnable_amplitude:# 预测幅度
            self.amplitude_predictor = nn.Linear(in_features, 1)
            with torch.no_grad():
                self.amplitude_predictor.weight.normal_(0, 0.1)
                self.amplitude_predictor.bias.fill_(1.0)
        else:
            self.register_parameter('fixed_amplitude', nn.Parameter(torch.tensor(1.0)))
        
        self.register_buffer('positions', torch.arange(out_features).float())
    def forward(self, x):
        batch_size = x.size(0)
        
        # 获取高斯参数（可变或固定）
        if self.learnable_mu:
            mu = torch.sigmoid(self.mu_predictor(x)) * self.out_features
        else:
            mu = self.fixed_mu.unsqueeze(0).expand(batch_size, 1)
        
        if self.learnable_sigma:
            sigma = F.softplus(self.sigma_predictor(x))
        else:
            sigma = self.fixed_sigma.unsqueeze(0).expand(batch_size, 1)
        
        if self.learnable_amplitude:
            amplitude = F.softplus(self.amplitude_predictor(x))
        else:
            amplitude = self.fixed_amplitude.unsqueeze(0).expand(batch_size, 1)
        
        # 生成高斯分布
        positions_expanded = self.positions.unsqueeze(0).expand(batch_size, -1)
        mu_expanded = mu.expand(-1, self.out_features)
        sigma_expanded = sigma.expand(-1, self.out_features)
        amplitude_expanded = amplitude.expand(-1, self.out_features)
        
        diff = positions_expanded - mu_expanded
        gaussian_output = amplitude_expanded * torch.exp(-diff**2 / (2 * sigma_expanded**2))
        
        return gaussian_output


#
"""支持多个高斯分布但输出仍是单峰的版本
model3 = MultiModalSingleGaussianFC(in_features, out_features, num_gaussians=2)"""
class MultiModalSingleGaussianFC(nn.Module):
    
    def __init__(self, in_features, out_features, num_gaussians=2):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.num_gaussians = num_gaussians
        
        # 为多个高斯分量预测参数
        self.mu_predictor = nn.Linear(in_features, num_gaussians)
        self.sigma_predictor = nn.Linear(in_features, num_gaussians)
        self.weight_predictor = nn.Linear(in_features, num_gaussians)  # 混合权重
        
        # 初始化
        with torch.no_grad():
            self.mu_predictor.weight.normal_(0, 0.1)
            self.mu_predictor.bias.uniform_(0, out_features)
            
            self.sigma_predictor.weight.normal_(0, 0.1)
            self.sigma_predictor.bias.fill_(out_features / 6)
            
            self.weight_predictor.weight.normal_(0, 0.1)
            self.weight_predictor.bias.fill_(1.0 / num_gaussians)
        
        self.register_buffer('positions', torch.arange(out_features).float())
    
    def forward(self, x):
        batch_size = x.size(0)
        
        # 预测多个高斯分量的参数
        mus = torch.sigmoid(self.mu_predictor(x)) * self.out_features  # (batch_size, num_gaussians)
        sigmas = F.softplus(self.sigma_predictor(x))  # (batch_size, num_gaussians)
        weights = F.softmax(self.weight_predictor(x), dim=1)  # (batch_size, num_gaussians)
        
        # 计算混合高斯分布
        positions_expanded = self.positions.unsqueeze(0).unsqueeze(2).expand(batch_size, -1, self.num_gaussians)
        mus_expanded = mus.unsqueeze(1).expand(-1, self.out_features, -1)
        sigmas_expanded = sigmas.unsqueeze(1).expand(-1, self.out_features, -1)
        weights_expanded = weights.unsqueeze(1).expand(-1, self.out_features, -1)
        
        # 计算每个高斯分量
        diff = positions_expanded - mus_expanded
        gaussians = torch.exp(-diff**2 / (2 * sigmas_expanded**2))
        
        # 加权求和得到最终输出
        output = (weights_expanded * gaussians).sum(dim=2)
        
        return output


#
"""
model3 = PeakedGaussianFC(in_features, out_features)
"""
class PeakedGaussianFC(nn.Module):
    def __init__(self, in_features, out_features):
        super().__init__()

        self.gaussian_fc = GaussianFC(in_features, out_features)
        self.temperature = nn.Parameter(torch.tensor(0.1))  # 温度参数，越小峰越尖锐
    
    def forward(self, x):
        output = self.gaussian_fc(x)
        # 使用softmax with temperature来增强主峰，抑制次峰
        peaked_output = torch.softmax(output / self.temperature, dim=-1)
        # 重新缩放到原来的幅度范围
        peaked_output = peaked_output * output.sum(dim=-1, keepdim=True)
        return peaked_output


#
# 后处理平滑
"""支持多种平滑方法:smooth_method='learnable
model3 = AdvancedSmoothedGaussianFC(in_features, out_features, 
                                       smooth_method='gaussian', 
                                       kernel_size=5, sigma=1.0, 
                                       smooth_strength=0.7)
                                       '"""
class AdvancedSmoothedGaussianFC(nn.Module):
    
    def __init__(self, in_features, out_features, 
                 smooth_method='gaussian', kernel_size=5, sigma=1.0, 
                 smooth_strength=0.7, learnable_params=True): #
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.smooth_method = smooth_method
        self.smooth_strength = smooth_strength
        
        # 原有的GaussianFC结构
        self.gaussian_fc = GaussianFC(in_features, out_features)
        
        # 根据平滑方法初始化不同的平滑器
        if smooth_method == 'gaussian':
            self._init_gaussian_smoother(kernel_size, sigma, learnable_params)
        elif smooth_method == 'average':
            self._init_average_smoother(kernel_size, learnable_params)
        elif smooth_method == 'learnable':
            self._init_learnable_smoother(kernel_size)
        else:
            raise ValueError(f"Unknown smooth_method: {smooth_method}")
    
    def _init_gaussian_smoother(self, kernel_size, sigma, learnable):
        """初始化高斯平滑器"""
        padding = kernel_size // 2
        self.smoother = nn.Conv1d(1, 1, kernel_size=kernel_size, padding=padding, bias=False)
        
        x = torch.linspace(-(kernel_size-1)/2, (kernel_size-1)/2, kernel_size)
        kernel = torch.exp(-x**2 / (2 * sigma**2))
        kernel = kernel / kernel.sum()
        
        with torch.no_grad():
            self.smoother.weight.data = kernel.view(1, 1, -1)
        
        self.smoother.weight.requires_grad = learnable
    
    def _init_average_smoother(self, kernel_size, learnable):
        """初始化均值平滑器"""
        padding = kernel_size // 2
        self.smoother = nn.Conv1d(1, 1, kernel_size=kernel_size, padding=padding, bias=False)
        
        kernel = torch.ones(kernel_size) / kernel_size  # 均值核
        
        with torch.no_grad():
            self.smoother.weight.data = kernel.view(1, 1, -1)
        
        self.smoother.weight.requires_grad = learnable
    
    def _init_learnable_smoother(self, kernel_size):
        """初始化完全可学习的平滑器"""
        padding = kernel_size // 2
        self.smoother = nn.Conv1d(1, 1, kernel_size=kernel_size, padding=padding, bias=False)
        # 使用默认的随机初始化
    
    def forward(self, x):
        # 原始输出
        original_output = self.gaussian_fc(x)
        
        # 平滑处理
        output_3d = original_output.unsqueeze(1)
        smoothed_3d = self.smoother(output_3d)
        smoothed_output = smoothed_3d.squeeze(1)
        
        # 混合原始输出和平滑输出
        final_output = (1 - self.smooth_strength) * original_output + \
                      self.smooth_strength * smoothed_output
        
        return final_output




# 使用示例
if __name__ == "__main__":
    # 标准ResNet18
    model1 = ResNet18(num_classes=10)
    x = torch.randn(2, 3, 224, 224)
    out1 = model1(x)
    print(f"ResNet18 输出形状: {out1.shape}")
    
    # SNN-ResNet18
    model2 = SNNResNet18(num_classes=10, beta=0.9, num_steps=10)
    spk, mem = model2(x)
    print(f"SNN-ResNet18 脉冲输出形状: {spk.shape}")
    print(f"SNN-ResNet18 膜电位输出形状: {mem.shape}")