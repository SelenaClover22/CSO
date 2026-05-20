"""Train a ResNet-18 on CIFAR-10 + ML-BadNet backdoor.

Uses the ML poison split written by attacks_crafting.py (suffix
'_<DPR>_<ratio>').

Saves under --model_dir (with suffix '_ml_<DPR>_<ratio>'):
  model_ml_<DPR>_<ratio>.pth              # checkpoint if attack succeeds
"""
import argparse
import os
import sys

import numpy as np
import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(_HERE))
from models.resnet import ResNet18


def parse_args():
    p = argparse.ArgumentParser(description="Train ResNet-18 with clean-label backdoor.")
    p.add_argument("--attack_dir", type=str, required=True)
    p.add_argument("--model_dir", type=str, required=True)
    p.add_argument("--DPR", type=float, default=0.01)
    p.add_argument("--ratio", type=float, default=2.0)
    p.add_argument("--data_root", type=str, default="./data")
    p.add_argument("--epochs", type=int, default=50)
    p.add_argument("--batch_size", type=int, default=32)
    p.add_argument("--test_batch_size", type=int, default=100)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--early_stop_asr", type=float, default=90.0,
                   help="Stop when both ACC and ASR reach this threshold.")
    p.add_argument("--early_stop_acc", type=float, default=90.0)
    return p.parse_args()


def lr_for(epoch, base):
    if epoch > 90:
        return base * 1e-3
    if epoch > 80:
        return base * 1e-2
    if epoch > 60:
        return base * 1e-1
    return base


def build_loaders(opt, suffix):
    tf_train = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
    ])
    tf_test = transforms.Compose([transforms.ToTensor()])

    trainset = torchvision.datasets.CIFAR10(
        root=opt.data_root, train=True, download=True, transform=tf_train,
    )
    testset = torchvision.datasets.CIFAR10(
        root=opt.data_root, train=False, download=True, transform=tf_test,
    )

    if not os.path.isdir(opt.attack_dir):
        sys.exit(f"attack_dir not found: {opt.attack_dir}")

    train_atk = torch.load(os.path.join(opt.attack_dir, f"train_attacks_ml_{suffix}"), weights_only=False)
    test_atk = torch.load(os.path.join(opt.attack_dir, "test_attacks"), weights_only=False)
    test_atk_ns = torch.load(os.path.join(opt.attack_dir, "test_attacks_nonsource"), weights_only=False)
    ind_train = torch.load(os.path.join(opt.attack_dir, f"ind_train_ml_{suffix}"), weights_only=False)

    image_dtype = trainset.data.dtype
    poison_imgs = np.rint(
        np.transpose(train_atk["image"].numpy() * 255, [0, 2, 3, 1])
    ).astype(image_dtype)
    trainset.data = np.concatenate((trainset.data, poison_imgs))
    trainset.targets = np.concatenate((trainset.targets, train_atk["label"]))
    trainset.data = np.delete(trainset.data, ind_train, axis=0)
    trainset.targets = np.delete(trainset.targets, ind_train, axis=0)

    trainloader = torch.utils.data.DataLoader(
        trainset, batch_size=opt.batch_size, shuffle=True, num_workers=2,
    )
    testloader = torch.utils.data.DataLoader(
        testset, batch_size=opt.test_batch_size, shuffle=False, num_workers=2,
    )
    atkloader = torch.utils.data.DataLoader(
        torch.utils.data.TensorDataset(test_atk["image"], test_atk["label"]),
        batch_size=opt.test_batch_size, shuffle=False, num_workers=2,
    )
    atkloader_ns = torch.utils.data.DataLoader(
        torch.utils.data.TensorDataset(test_atk_ns["image"], test_atk_ns["label"]),
        batch_size=opt.test_batch_size, shuffle=False, num_workers=2,
    )
    return trainloader, testloader, atkloader, atkloader_ns


def train_one_epoch(net, loader, criterion, optimizer, device):
    net.train()
    correct = total = 0
    for inputs, targets in loader:
        inputs, targets = inputs.to(device), targets.to(device)
        optimizer.zero_grad()
        outputs = net(inputs)
        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()
        total += targets.size(0)
        correct += outputs.argmax(1).eq(targets).sum().item()
    return 100.0 * correct / total


@torch.no_grad()
def evaluate(net, loader, device):
    net.eval()
    correct = total = 0
    for inputs, targets in loader:
        inputs, targets = inputs.to(device), targets.to(device)
        outputs = net(inputs)
        total += targets.size(0)
        correct += outputs.argmax(1).eq(targets).sum().item()
    return 100.0 * correct / total


def main():
    opt = parse_args()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    os.makedirs(opt.model_dir, exist_ok=True)
    suffix = f"{opt.DPR:g}_{opt.ratio:g}"
    print(f"-------- {opt.attack_dir} -> {opt.model_dir} (ml_{suffix}) --------")

    trainloader, testloader, atkloader, atkloader_ns = build_loaders(opt, suffix)
    net = ResNet18(num_classes=10).to(device)
    criterion = nn.CrossEntropyLoss()

    best_model = os.path.join(opt.model_dir, f"model_ml_{suffix}.pth")

    success = False
    for epoch in range(opt.epochs):
        lr = lr_for(epoch, opt.lr)
        optimizer = torch.optim.Adam(net.parameters(), lr=lr)
        train_acc = train_one_epoch(net, trainloader, criterion, optimizer, device)
        acc = evaluate(net, testloader, device)
        asr = evaluate(net, atkloader, device)
        cd = evaluate(net, atkloader_ns, device)
        print(f"epoch {epoch:3d}  lr={lr:.1e}  train_acc={train_acc:.2f}  "
              f"test_acc={acc:.2f}  asr={asr:.2f}  collateral_damage={cd:.2f}")

        if asr >= opt.early_stop_asr and acc >= opt.early_stop_acc:
            print("Reached early-stop thresholds.")
            torch.save(net.state_dict(), best_model)
            success = True
            break

    if not success:
        print("please craft again because this one does not succeed")


if __name__ == "__main__":
    main()
