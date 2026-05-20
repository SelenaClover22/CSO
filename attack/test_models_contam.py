import argparse
import os
import sys

import numpy as np
import torch
import torchvision
import torchvision.transforms as transforms

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(_HERE))
from models.resnet import ResNet18
from utils import mask_craft, embed_backdoor


NUM_OF_ATTACKS = 3000


def parse_args():
    p = argparse.ArgumentParser(description="Evaluate a backdoored CIFAR-10 model.")
    p.add_argument("--attack_dir", type=str, required=True)
    p.add_argument("--model_dir", type=str, required=True)
    p.add_argument("--ckpt_name", type=str, default="model.pth",
                   help="Checkpoint filename inside --model_dir.")
    p.add_argument("--data_root", type=str, default="./data")
    p.add_argument("--batch_size", type=int, default=100)
    return p.parse_args()


def read_attack_info(path):
    with open(path) as f:
        line = f.readline().strip()
    # "source class: SC. target class: TC"
    sc = int(line.split(":")[1].split(".")[0].strip())
    tc = int(line.split(":")[-1].strip())
    return sc, tc


@torch.no_grad()
def evaluate(net, loader, device):
    correct = total = 0
    for inputs, targets in loader:
        inputs, targets = inputs.to(device), targets.to(device)
        total += targets.size(0)
        correct += net(inputs).argmax(1).eq(targets).sum().item()
    return 100.0 * correct / total


@torch.no_grad()
def per_class_asr(net, loader, gt_labels, num_classes, device):
    correct = [0] * num_classes
    total = [0] * num_classes
    offset = 0
    batch_size = loader.batch_size
    for batch_idx, (inputs, targets) in enumerate(loader):
        inputs, targets = inputs.to(device), targets.to(device)
        hit = net(inputs).argmax(1).eq(targets)
        for j in range(targets.size(0)):
            true_label = int(gt_labels[offset + j])
            correct[true_label] += int(hit[j].item())
            total[true_label] += 1
        offset += targets.size(0)
    return correct, total


def main():
    opt = parse_args()
    device = "cuda" if torch.cuda.is_available() else "cpu"

    SC, TC = read_attack_info(os.path.join(opt.attack_dir, "attack_info.txt"))
    print(f"source class: {SC}, target class: {TC}")

    tf = transforms.Compose([transforms.ToTensor()])
    testset = torchvision.datasets.CIFAR10(
        root=opt.data_root, train=False, download=True, transform=tf,
    )
    testloader = torch.utils.data.DataLoader(
        testset, batch_size=opt.batch_size, shuffle=False, num_workers=2,
    )

    # Trigger pattern + on-the-fly non-source attack set.
    pattern = torch.load(os.path.join(opt.attack_dir, "pattern"), weights_only=False)
    mask = mask_craft(pattern)
    ind_ns = [i for i, y in enumerate(testset.targets) if y != SC and y != TC]
    ind_ns = np.random.choice(ind_ns, NUM_OF_ATTACKS, replace=False)
    ns_imgs, ns_labels = [], []
    ns_gt = []
    for i in ind_ns:
        img, lbl = testset[i]
        ns_imgs.append(embed_backdoor(img, pattern, mask).unsqueeze(0))
        ns_labels.append(torch.tensor([TC], dtype=torch.long))
        ns_gt.append(lbl)
    ns_imgs = torch.cat(ns_imgs, dim=0)
    ns_labels = torch.cat(ns_labels, dim=0)
    ns_gt = torch.tensor(ns_gt, dtype=torch.long)

    atkloader_ns = torch.utils.data.DataLoader(
        torch.utils.data.TensorDataset(ns_imgs, ns_labels),
        batch_size=opt.batch_size, shuffle=False, num_workers=2,
    )

    # Saved source-class trigger images (label==TC).
    test_atk = torch.load(os.path.join(opt.attack_dir, "test_attacks"), weights_only=False)
    atkloader = torch.utils.data.DataLoader(
        torch.utils.data.TensorDataset(test_atk["image"], test_atk["label"]),
        batch_size=opt.batch_size, shuffle=False, num_workers=2,
    )

    # Load model.
    ckpt_path = os.path.join(opt.model_dir, opt.ckpt_name)
    if not os.path.isfile(ckpt_path):
        sys.exit(f"checkpoint not found: {ckpt_path}")
    model = ResNet18(num_classes=10).to(device).eval()
    state = torch.load(ckpt_path, map_location=device, weights_only=False)
    if isinstance(state, dict):
        for k in ("netC", "model", "net", "state_dict"):
            if k in state and isinstance(state[k], dict):
                state = state[k]
                break
    if any(k.startswith("module.") for k in state.keys()):
        state = {k[len("module."):]: v for k, v in state.items()}
    model.load_state_dict(state)

    print(f"Test ACC:         {evaluate(model, testloader, device):.3f}")
    print(f"Attack success:   {evaluate(model, atkloader, device):.3f}")

    correct, total = per_class_asr(model, atkloader_ns, ns_gt, 10, device)
    for c in range(10):
        if c == SC or c == TC:
            continue
        if total[c] > 0:
            print(f"  ASR class {c}: {100.0 * correct[c] / total[c]:.2f}%")
        else:
            print(f"  ASR class {c}: no samples")
    overall_correct = sum(correct[c] for c in range(10) if c != SC and c != TC)
    overall_total = sum(total[c] for c in range(10) if c != SC and c != TC)
    print(f"Collateral Damage: {100.0 * overall_correct / overall_total:.3f}")


if __name__ == "__main__":
    main()
