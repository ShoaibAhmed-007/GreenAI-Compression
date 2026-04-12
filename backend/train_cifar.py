"""
Modular CIFAR-10 training script focused on accuracy + efficiency (Green AI).
Features:
- Model adapters for ResNet18, DenseNet121, MobileNetV2 adjusted for 32x32 inputs
- Data augmentation: RandomCrop(32,padding=4), RandomHorizontalFlip, optional ColorJitter, Normalize
- Optional Cutout and MixUp
- SGD with momentum (0.9) and weight decay 5e-4
- CosineAnnealingLR scheduler (T_max = epochs)
- Label smoothing (0.1)
- Classifier Dropout for better confidence/generalization
- Modular: `get_model(name, ...)` to plug models easily

Usage examples:
python backend/train_cifar.py --model resnet18 --epochs 200 --batch-size 128

"""

import argparse
import os
import time
from collections import OrderedDict

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader
import torchvision
import torchvision.transforms as transforms
from torchvision import models
from torchvision.models import DenseNet121_Weights, MobileNet_V2_Weights, ResNet18_Weights


# ---------------------- Utilities ----------------------
class LabelSmoothingCrossEntropy(nn.Module):
    def __init__(self, smoothing=0.1):
        super().__init__()
        assert 0.0 <= smoothing < 1.0
        self.smoothing = smoothing

    def forward(self, x, target):
        # x: logits (N, C), target: (N,)
        n_classes = x.size(1)
        log_preds = F.log_softmax(x, dim=1)
        with torch.no_grad():
            true_dist = torch.zeros_like(x)
            true_dist.fill_(self.smoothing / (n_classes - 1))
            true_dist.scatter_(1, target.data.unsqueeze(1), 1.0 - self.smoothing)
        return torch.mean(torch.sum(-true_dist * log_preds, dim=1))


def accuracy(output, target):
    with torch.no_grad():
        pred = output.argmax(dim=1)
        correct = pred.eq(target).sum().item()
        return correct / target.size(0)


# ---------------------- Augmentations ----------------------
class Cutout(object):
    def __init__(self, length):
        self.length = length

    def __call__(self, img):
        h, w = img.size(1), img.size(2)
        mask = torch.ones((h, w), dtype=torch.float32)
        y = torch.randint(h, (1,)).item()
        x = torch.randint(w, (1,)).item()

        y1 = max(0, y - self.length // 2)
        y2 = min(h, y + self.length // 2)
        x1 = max(0, x - self.length // 2)
        x2 = min(w, x + self.length // 2)

        mask[y1:y2, x1:x2] = 0.
        mask = mask.expand(3, -1, -1)
        return img * mask


# ---------------------- MixUp ----------------------
def mixup_data(x, y, alpha=1.0, device='cpu'):
    if alpha <= 0:
        return x, y, None, 1.0
    lam = np.random.beta(alpha, alpha)
    batch_size = x.size()[0]
    index = torch.randperm(batch_size).to(device)
    mixed_x = lam * x + (1 - lam) * x[index, :]
    y_a, y_b = y, y[index]
    return mixed_x, y_a, y_b, lam


# ---------------------- Model adapters ----------------------
def get_model(name='resnet18', pretrained=False, num_classes=10,
              dropout_p=0.35, adapt_stem_for_cifar=False):
    name = name.lower()
    if name == 'resnet18':
        weights = ResNet18_Weights.DEFAULT if pretrained else None
        model = models.resnet18(weights=weights)
        # For native 32x32 training, a CIFAR-friendly stem helps preserve detail.
        if adapt_stem_for_cifar:
            model.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
            model.maxpool = nn.Identity()
        model.fc = nn.Sequential(
            nn.Dropout(p=dropout_p),
            nn.Linear(model.fc.in_features, num_classes),
        )
        return model

    if name == 'densenet121' or name == 'densenet':
        weights = DenseNet121_Weights.DEFAULT if pretrained else None
        model = models.densenet121(weights=weights)
        if adapt_stem_for_cifar:
            # Replace initial conv with 3x3 and remove pool for 32x32 inputs.
            model.features.conv0 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
            if hasattr(model.features, 'pool0'):
                model.features.pool0 = nn.Identity()
        model.classifier = nn.Sequential(
            nn.Dropout(p=dropout_p),
            nn.Linear(model.classifier.in_features, num_classes),
        )
        return model

    if name == 'mobilenetv2' or name == 'mobilenet_v2':
        weights = MobileNet_V2_Weights.DEFAULT if pretrained else None
        model = models.mobilenet_v2(weights=weights)
        if adapt_stem_for_cifar:
            # Change first conv stride from 2 to 1 to preserve 32x32.
            try:
                model.features[0][0] = nn.Conv2d(3, 32, kernel_size=3, stride=1, padding=1, bias=False)
            except Exception:
                try:
                    model.features[0] = nn.Conv2d(3, 32, kernel_size=3, stride=1, padding=1, bias=False)
                except Exception:
                    pass
        # Replace classifier
        if isinstance(model.classifier, nn.Sequential):
            if len(model.classifier) > 0 and isinstance(model.classifier[0], nn.Dropout):
                model.classifier[0] = nn.Dropout(p=dropout_p)
            model.classifier[1] = nn.Linear(model.classifier[1].in_features, num_classes)
        else:
            model.classifier = nn.Sequential(
                nn.Dropout(p=dropout_p),
                nn.Linear(model.last_channel, num_classes),
            )
        return model

    raise ValueError(f"Unsupported model: {name}")


# ---------------------- Training / Validation ----------------------
import numpy as np


def train_one_epoch(model, loader, optimizer, criterion, device, epoch, scaler=None, mixup_alpha=0.0):
    model.train()
    running_loss = 0.0
    running_acc = 0.0
    total = 0

    for i, (images, targets) in enumerate(loader):
        images = images.to(device)
        targets = targets.to(device)

        if mixup_alpha > 0:
            images, targets_a, targets_b, lam = mixup_data(images, targets, mixup_alpha, device=device)
        else:
            lam = 1.0
            targets_a = targets
            targets_b = None

        optimizer.zero_grad()

        if scaler is not None:
            with torch.cuda.amp.autocast():
                outputs = model(images)
                if targets_b is not None:
                    loss = lam * criterion(outputs, targets_a) + (1 - lam) * criterion(outputs, targets_b)
                else:
                    loss = criterion(outputs, targets_a)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            outputs = model(images)
            if targets_b is not None:
                loss = lam * criterion(outputs, targets_a) + (1 - lam) * criterion(outputs, targets_b)
            else:
                loss = criterion(outputs, targets_a)
            loss.backward()
            optimizer.step()

        bs = targets.size(0)
        running_loss += loss.item() * bs
        if targets_b is not None:
            # For mixup, accuracy is not strictly meaningful, but we can compute on targets_a
            running_acc += accuracy(outputs, targets_a) * bs
        else:
            running_acc += accuracy(outputs, targets) * bs
        total += bs

    return running_loss / total, running_acc / total


def validate(model, loader, criterion, device):
    model.eval()
    val_loss = 0.0
    val_acc = 0.0
    total = 0
    with torch.no_grad():
        for images, targets in loader:
            images = images.to(device)
            targets = targets.to(device)
            outputs = model(images)
            loss = criterion(outputs, targets)
            bs = targets.size(0)
            val_loss += loss.item() * bs
            val_acc += accuracy(outputs, targets) * bs
            total += bs
    return val_loss / total, val_acc / total


# ---------------------- Main / CLI ----------------------

def parse_args():
    parser = argparse.ArgumentParser(description='CIFAR-10 training (Green AI friendly)')
    parser.add_argument('--model', default='resnet18', choices=['resnet18', 'densenet121', 'mobilenetv2'], help='model name')
    parser.add_argument('--epochs', default=120, type=int)
    parser.add_argument('--batch-size', default=128, type=int)
    parser.add_argument('--lr', default=None, type=float,
                        help='learning rate (default: 0.01 pretrained, 0.1 scratch)')
    parser.add_argument('--momentum', default=0.9, type=float)
    parser.add_argument('--weight-decay', default=5e-4, type=float)
    parser.add_argument('--input-size', default=224, type=int,
                        help='input size (224 for pretrained ImageNet backbones, 32 for scratch CIFAR stem)')
    parser.add_argument('--dropout', default=0.35, type=float, help='classifier dropout probability')
    parser.add_argument('--color-jitter', default=0.15, type=float, help='color jitter strength (0 to disable)')
    parser.add_argument('--smoothing', default=0.1, type=float)
    parser.add_argument('--mixup', default=0.0, type=float, help='mixup alpha (0 to disable)')
    parser.add_argument('--cutout-length', default=0, type=int, help='cutout length (0 to disable)')
    parser.add_argument('--data-dir', default='./data', type=str)
    parser.add_argument('--workers', default=4, type=int)
    parser.add_argument('--save-dir', default='./checkpoints', type=str)
    parser.add_argument('--use-amp', action='store_true', help='use mixed precision')
    parser.add_argument('--no-pretrained', dest='pretrained', action='store_false')
    parser.set_defaults(pretrained=True)
    return parser.parse_args()


def main():
    args = parse_args()
    os.makedirs(args.save_dir, exist_ok=True)

    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    if args.input_size < 32:
        args.input_size = 32
    if args.pretrained and args.input_size < 96:
        print('Using pretrained backbone: overriding input size to 224 for stable transfer learning.')
        args.input_size = 224

    # For tiny native inputs, use a CIFAR-friendly stem. For resized 224 inputs, keep ImageNet stem.
    adapt_stem_for_cifar = args.input_size <= 64

    if args.lr is None:
        args.lr = 0.01 if args.pretrained else 0.1

    print(
        f"Config | model={args.model} pretrained={args.pretrained} input={args.input_size} "
        f"epochs={args.epochs} batch={args.batch_size} lr={args.lr}"
    )

    # Data transforms
    mean = (0.4914, 0.4822, 0.4465)
    std = (0.2470, 0.2435, 0.2616)

    train_transforms = [
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
    ]
    if args.input_size > 32:
        train_transforms.append(
            transforms.Resize(args.input_size, interpolation=transforms.InterpolationMode.BICUBIC)
        )
    if args.color_jitter > 0:
        cj = float(args.color_jitter)
        train_transforms.append(
            transforms.ColorJitter(brightness=cj, contrast=cj, saturation=max(cj * 0.75, 0.0), hue=min(cj * 0.2, 0.1))
        )
    train_transforms.extend([transforms.ToTensor(), transforms.Normalize(mean, std)])
    if args.cutout_length > 0:
        train_transforms.append(Cutout(args.cutout_length))
    train_transform = transforms.Compose(train_transforms)

    test_transform = transforms.Compose([
        transforms.Resize(args.input_size, interpolation=transforms.InterpolationMode.BICUBIC),
        transforms.ToTensor(),
        transforms.Normalize(mean, std),
    ])

    # Datasets
    train_set = torchvision.datasets.CIFAR10(root=args.data_dir, train=True, download=True, transform=train_transform)
    test_set = torchvision.datasets.CIFAR10(root=args.data_dir, train=False, download=True, transform=test_transform)

    train_loader = DataLoader(train_set, batch_size=args.batch_size, shuffle=True, num_workers=args.workers, pin_memory=True)
    test_loader = DataLoader(test_set, batch_size=args.batch_size, shuffle=False, num_workers=args.workers, pin_memory=True)

    # Model
    model = get_model(
        args.model,
        pretrained=args.pretrained,
        num_classes=10,
        dropout_p=max(0.0, min(float(args.dropout), 0.8)),
        adapt_stem_for_cifar=adapt_stem_for_cifar,
    )
    model = model.to(device)

    # Criterion with label smoothing
    criterion = LabelSmoothingCrossEntropy(smoothing=args.smoothing)

    # Optimizer and scheduler
    optimizer = optim.SGD(
        model.parameters(),
        lr=args.lr,
        momentum=args.momentum,
        weight_decay=args.weight_decay,
        nesterov=True,
    )
    scheduler = optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=args.epochs,
        eta_min=max(args.lr * 0.01, 1e-6),
    )

    scaler = torch.cuda.amp.GradScaler() if (args.use_amp and device == 'cuda') else None

    best_acc = 0.0
    for epoch in range(1, args.epochs + 1):
        start = time.time()
        train_loss, train_acc = train_one_epoch(model, train_loader, optimizer, criterion, device, epoch, scaler=scaler, mixup_alpha=args.mixup)
        val_loss, val_acc = validate(model, test_loader, criterion, device)
        scheduler.step()

        elapsed = time.time() - start
        print(f'Epoch {epoch:03d}/{args.epochs}  Time {elapsed:.1f}s  TrainLoss {train_loss:.4f} TrainAcc {train_acc:.4f}  ValLoss {val_loss:.4f} ValAcc {val_acc:.4f}')

        # Save best
        if val_acc > best_acc:
            best_acc = val_acc
            save_path = os.path.join(args.save_dir, f'{args.model}_best.pth')
            torch.save({'epoch': epoch, 'model_state': model.state_dict(), 'best_acc': best_acc, 'optimizer': optimizer.state_dict()}, save_path)

    print(f'Best validation accuracy: {best_acc:.4f}')


if __name__ == '__main__':
    main()
