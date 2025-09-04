from dataset_loader import get_cifar10_dataloaders, get_dataloader
from dataset_loader import get_cifar100_dataloaders
from dataset_loader import get_imagenet_dataloaders, get_fgnet_dataloaders, get_hic_dataloaders

from torchvision.models import resnet18, ResNet18_Weights
from torch.utils.data import DataLoader
from torchvision import transforms
from time import time
import torch
import losses, dataset_loader
from models import MLP
from model import *

import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--dataset', required=True)
parser.add_argument('--loss', required=True)
parser.add_argument('--rep', type=int, required=True)
parser.add_argument('--output', default='output/model.pth')
parser.add_argument('--lamda', type=float)
parser.add_argument('--batchsize', type=int, default=32)
#parser.add_argument('--classes_num', type=int, default=70)
parser.add_argument('--lr', default=1e-3, type=float)
parser.add_argument('--epochs', default=100, type=int)
parser.add_argument('--model', default="resnet18_weights", type=str)
parser.add_argument('--use_pre_net', action='store_true', 
                    help="Use pretrained network if set") #python train.py --use_pre_net
parser.add_argument('--fc_layer_name', default="GaussianFC", type=str)    
parser.add_argument('--use_leaky_gaussianfc', default=2, type=int)    
parser.add_argument('--learnable_params', type=bool, default=False)
parser.add_argument('--use_original_loss', type=bool, default=False)

args = parser.parse_args()



device = 'cuda' if torch.cuda.is_available() else 'cpu'

if args.dataset == 'cifar10':
    train_loader, test_loader = get_cifar10_dataloaders(
        batch_size=args.batchsize,
        num_workers=4,
        augment=True,      # 使用数据增强
        normalize=True     # 标准化
    )
    num_classes = 10
elif args.dataset == 'cifar100':
    # 2. CIFAR-100（细粒度100类
    train_loader, test_loader = get_cifar100_dataloaders(
        batch_size=args.batchsize,
        num_workers=4,
        augment=True,
        normalize=True,
        use_coarse_labels=False  # False=100类, True=20类
    )
    num_classes = 100
elif args.dataset == 'imagenet':
    # 3. ImageNet
    train_loader, test_loader = get_imagenet_dataloaders(
        batch_size=args.batchsize,
        num_workers=8,
        image_size=224,    # ResNet标准尺寸
        augment=True,
        normalize=True
    )
elif args.dataset == 'FGNET':
    train_loader, test_loader, num_classes = get_dataloader(
        'fgnet',
        batch_size=args.batchsize,
        num_workers=4,
        image_size=256,
        augment=True,
        normalize=True
        )
elif args.dataset == 'HIC':
    train_loader, test_loader, num_classes = get_dataloader(
        'hic',
        batch_size=args.batchsize,
        num_workers=4,
        image_size=224,
        augment=True,
        normalize=True
        )




if args.lamda is None:
    loss_fn = getattr(losses, args.loss)(num_classes)
else:
    try:
        loss_fn = getattr(losses, args.loss)(num_classes, args.lamda)
    except:
        loss_fn = getattr(losses, args.loss)(num_classes)



from omegaconf import OmegaConf 
cfg = OmegaConf.create({
    'loss': args.loss,  #  
    'model': {
        'neuron': {
            'beta': 0.9
        },
        'time_steps': 10,
        'layer': {
            'dropout_rate': 0.5
        }
    }
})



def get_fc_layer(use_leaky_gaussianfc,in_features, out_features,layer_name =None):
    """根据层名称返回对应的层实例"""
    layer_map = {
        'GaussianFC': GaussianFC,
        'GaussianFC_uni': GaussianFC_uni,
        'FlexibleSingleGaussianFC': FlexibleSingleGaussianFC,
        'MultiModalSingleGaussianFC': MultiModalSingleGaussianFC,
        'PeakedGaussianFC': PeakedGaussianFC,
        'AdvancedSmoothedGaussianFC': AdvancedSmoothedGaussianFC,
    }
    
    layers_with_learnable_params = {
        'AdvancedSmoothedGaussianFC'
    }

    def create_layer_instance(layer_class, in_feat, out_feat):
        """创建层实例，根据层类型决定是否传递 learnable_params"""
        if layer_name in layers_with_learnable_params and learnable_params is not None:
            return layer_class(in_feat, out_feat, learnable_params=learnable_params)
        else:
            return layer_class(in_feat, out_feat)

    if use_leaky_gaussianfc == 0:
        fc = torch.nn.Linear(in_features, out_features)
    elif use_leaky_gaussianfc == 1:
        layer_class = layer_map.get(layer_name, torch.nn.Linear)
        if layer_class == torch.nn.Linear:
            fc = layer_class(in_features, out_features)
        else:
            fc = create_layer_instance(layer_class, in_features, out_features)
    elif use_leaky_gaussianfc == 2:
        layer_class = layer_map.get(layer_name, torch.nn.Linear)
        linear_layer = torch.nn.Linear(in_features, out_features)
        
        if layer_class == torch.nn.Linear:
            gaussian_layer = layer_class(out_features, out_features)
        else:
            gaussian_layer = create_layer_instance(layer_class, out_features, out_features)
        
        fc = nn.Sequential(linear_layer, gaussian_layer)

    return fc



out_features = loss_fn.how_many_outputs()  # learnable_params

if args.use_pre_net:
    if args.dataset == 'tabular':
        model = MLP(ds[0][0].shape[0], 128, out_features)
        args.epochs = 1000
    elif args.model == "resnet18_weights":
        #model = resnet18(weights=ResNet18_Weights.DEFAULT)
        model = resnet18(weights=ResNet18_Weights.IMAGENET1K_V2) #ResNet18_Weights.IMAGENET1K_V2 ResNet18_Weights.DEFAULT
        model.fc = get_fc_layer(0, 512, num_classes)
    elif args.model == "resnet18_GaussianFC_weights":
        model = resnet18(weights=ResNet18_Weights.IMAGENET1K_V2)
        model.fc = get_fc_layer(args.use_leaky_gaussianfc, 512, num_classes, args.fc_layer_name)
    elif args.model == "resnet18_GaussianFC":
        model = resnet18(weights=None)
        model.fc = get_fc_layer(args.use_leaky_gaussianfc, 512, num_classes, args.fc_layer_name)
    elif args.model == "resnet18":
        model = resnet18(weights=None)
        model.fc = get_fc_layer(0, 512, num_classes)
else:
    from snn_models import SNNResNet18,ResNet18
    # 使用
    if args.model == "resnet18":
        if args.loss == 'POM':
            model = SNNResNet18(cfg, num_classes)
            model.fc = get_fc_layer(0, 512, 1) #仅使用全连接层            
        else:
            model = SNNResNet18(cfg, num_classes)
            model.fc = get_fc_layer(0, 512, num_classes) #仅使用全连接层
    elif args.model == "resnet18_GaussianFC":
        model = ResNet18(cfg, num_classes)
        model.fc = get_fc_layer(args.use_leaky_gaussianfc, 512, num_classes, args.fc_layer_name)
    elif args.model == "snn_resnet18":
        if args.loss == 'POM':
            model = SNNResNet18(cfg, num_classes)
            model.fc = get_fc_layer(0, 512, 1) #仅使用全连接层            
        else:
            model = SNNResNet18(cfg, num_classes)
            model.fc = get_fc_layer(0, 512, num_classes) #仅使用全连接层
    elif args.model == "snn_resnet18_GaussianFC":
        model = SNNResNet18(cfg, num_classes)
        model.fc = get_fc_layer(args.use_leaky_gaussianfc,  512, num_classes, args.fc_layer_name)



loss_fn.to(device)
model = model.to(device)
model.loss_fn = loss_fn

opt = torch.optim.Adam(model.parameters(), args.lr)


train_time = 0
test_time = 0
Best_test_acc = []

#early stop
best_test_loss = float('inf')
patience_counter = 0
patience = 20

for epoch in range(args.epochs):
    model.train()
    if epoch % 30 == 0:
        print(f'* Epoch {epoch+1}/{args.epochs}')
    tic = time()
    avg_loss = 0
    correct = 0
    total = 0
    for X, Y in train_loader:
        X = X.to(device)
        Y = Y.to(device)
        Yhat = model(X)
        _, predicted = torch.max(Yhat, 1)
        total += Y.size(0)
        correct += (predicted == Y).sum().item()
        if args.model in ["resnet18_GaussianFC", "snn_resnet18_GaussianFC"] and args.use_original_loss is not true:
            probs = F.softmax(Yhat, dim=1)
            expected_class = torch.sum(probs * torch.arange(num_classes, dtype=torch.float,device=probs.device), dim=1)
            loss_value = torch.abs(expected_class - Y.float()).mean()
        else:
            loss_value = loss_fn(Yhat, Y).mean()
        opt.zero_grad()
        loss_value.backward()
        opt.step()
        avg_loss += float(loss_value) / len(train_loader)

    toc = time()
    accuracy = 100 * correct / total
    if epoch % 30 == 0:
        print(f'- {toc-tic:.0f}s - train_Loss: {avg_loss:.4f} - train_Acc: {accuracy:.2f}%')
    train_time += toc-tic
#################################
    model.eval()  # 设置为评估模式
    test_tic = time()
    test_loss = 0
    test_correct = 0
    test_total = 0
    with torch.no_grad():  # 测试时不需要计算梯度
        for X, Y in test_loader:
            X = X.to(device)
            Y = Y.to(device)
            # 前向传播
            Yhat = model(X)
            #print ('Yhat.shape',Yhat.shape,Yhat[0],Y)
            if args.model in ["resnet18_GaussianFC", "snn_resnet18_GaussianFC"]:
                probs = F.softmax(Yhat, dim=1)
                expected_class = torch.sum(probs * torch.arange(num_classes, dtype=torch.float,device=probs.device), dim=1)
                loss_value = torch.abs(expected_class - Y.float()).mean()
            else:
                loss_value = loss_fn(Yhat, Y).mean()
            # 统计测试指标
            test_loss += float(loss_value) / len(test_loader)
            _, predicted = torch.max(Yhat, 1)
            test_total += Y.size(0)
            test_correct += (predicted == Y).sum().item()
    test_toc = time()
    test_accuracy = 100 * test_correct / test_total
    Best_test_acc.append(test_accuracy)
    test_time += test_toc - test_tic
    if epoch % 30 == 0:
        print(f'- {test_toc-test_tic:.0f}s - test_Loss: {test_loss:.4f} - test_Acc: {test_accuracy:.2f}%')

    #早停
    if test_loss < best_test_loss:
        best_test_loss = test_loss
        patience_counter = 0
    else:
        patience_counter += 1
        if patience_counter >= patience:
            print(f'Early stopping at epoch {epoch+1}')
            break

time = test_time + train_time
#model.train_time = train_time
#torch.save(model.cpu(), args.output)

Best_ta = max(Best_test_acc)

print(f'dataset: {args.dataset}, loss_fn: {args.loss}, non_pre_weighted\n'
      f'batch_size: {args.batchsize}, lr: {args.lr}, epochs: {args.epochs}\n'
      f'time: {time:.2f}s, model: {args.model}\n'
      f'lamda: {args.lamda if args.lamda is not None else "None"},use_original_loss: {args.use_original_loss}\n'
      f'fc_layer_name: {args.fc_layer_name}, use_leaky_gaussianfc:{args.use_leaky_gaussianfc}\n'
      f'batch_size: {args.batchsize},Best_test_acc: {Best_ta}, Best_test_acc_index: {Best_test_acc.index(Best_ta)}\n')