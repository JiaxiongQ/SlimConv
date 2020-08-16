
import time

import torch.nn.parallel
import torch.backends.cudnn as cudnn

import torch.optim
import torch.utils.data
import torch.utils.data.distributed
import torchvision.transforms as transforms
import torchvision.datasets as datasets
import os
import argparse

# from models import *
from .SC-ResNet import *
from utils import progress_bar

parser = argparse.ArgumentParser(description='PyTorch ImageNet Test')
parser.add_argument('--resume', '-r', action='store_true', help='resume from checkpoint')
args = parser.parse_args()

device = 'cuda' if torch.cuda.is_available() else 'cpu'
best_acc = 0  # best test accuracy
start_epoch = 0  # start from epoch 0 or last checkpoint epoch

# Data
print('==> Preparing data..')
valdir = os.path.join('../data/', 'val')

normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225])

val_loader = torch.utils.data.DataLoader(
    datasets.ImageFolder(valdir, transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        normalize,
    ])),
    batch_size=256, shuffle=False,
    num_workers=4, pin_memory=True)

# exit(0)
# Model
print('==> Building model..')

net = resnet50()

if device == 'cuda':
    net = torch.nn.DataParallel(net).cuda()
    cudnn.benchmark = True

if args.resume:
    # Load checkpoint.
    print('==> Resuming from checkpoint..')
    checkpoint = torch.load('./')
    best = checkpoint['best_prec1']
    print(best)
    net.load_state_dict(checkpoint['state_dict'])

criterion = nn.CrossEntropyLoss()

def test():
    global best_acc
    net.eval()
    test_loss = 0
    correct = 0
    total = 0
    latency=0
    with torch.no_grad():
        for batch_idx, (inputs, targets) in enumerate(val_loader):
            inputs, targets = inputs.to(device), targets.to(device)
			
            time_s = time.time()
            outputs = net(inputs)
            torch.cuda.synchronize()
            latency += time.time() - time_s
			
            loss = criterion(outputs, targets)

            test_loss += loss.item()
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()

            progress_bar(batch_idx, len(val_loader), 'Loss: %.3f | Acc: %.3f%% (%d/%d)'
                % (test_loss/(batch_idx+1), 100.*correct/total, correct, total))
    avg_time = latency
    print(avg_time)
def test2():
    global best_acc
    net.eval()
    test_loss = 0
    correct = 0
    total = 0
    with torch.no_grad():
        for batch_idx, (inputs, targets) in enumerate(val_loader):
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = net(inputs)

            _, predicted = outputs.topk(5,1,True,True)

            total += targets.size(0)
            predicted = predicted.t()
            ct = predicted.eq(targets.view(1,-1).expand_as(predicted))

            correct += ct[:5].view(-1).float().sum(0).item()

            progress_bar(batch_idx, len(val_loader), 'Loss: %.3f | Acc: %.3f%% (%d/%d)'
                % (test_loss/(batch_idx+1), 100.*correct/total, correct, total))
test()
#test2()
