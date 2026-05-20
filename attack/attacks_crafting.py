"""Craft 1-to-1 BadNet backdoor images for one attack instance.

Both original BadNet and Mixed-Label(ML)-BadNet backdoor images are crafted.


Outputs (saved under --out_dir):
  train_attacks                       # dirty-label poison set (label==TC)
  train_attacks_ml_<DPR>_<ratio>      # clean-label poison set
  ind_train                           # source-class indices replaced by the poisons
  ind_train_ml_<DPR>_<ratio>          # all indices used in the ML set
  test_attacks                        # source-class test images with trigger, labels flipped to TC
  test_attacks_nonsource              # non-source / non-target images with trigger, labels not flipped to TC
  pattern                             # the trigger tensor
  attack_info.txt                     # 'source class: SC. target class: TC'
"""
import argparse
import os
import random

import numpy as np
import torch
import torchvision
import torchvision.transforms as transforms

from utils import pattern_craft, mask_craft, embed_backdoor


NUM_CLASSES = 10
ONE_CLASS_NUM = 5000


def parse_args():
    p = argparse.ArgumentParser(description="Craft BadNet backdoor images.")
    p.add_argument("--out_dir", type=str, required=True,
                   help="Output directory for crafted backdoor images.")
    p.add_argument("--DPR", type=float, default=0.01,
                   help="Dirty-label poison rate (fraction of source class poisoned).")
    p.add_argument("--ratio", type=float, default=2.0,
                   help="OPR/DPR ratio for clean-label poisoning.")
    p.add_argument("--data_root", type=str, default="./data")
    return p.parse_args()


def _make_poison_split(indices, items_getter, label_resolver, pattern, mask):
    imgs, lbls = [], []
    for i in indices:
        img, _ = items_getter(i)
        imgs.append(embed_backdoor(img, pattern, mask).unsqueeze(0))
        lbls.append(torch.tensor([label_resolver(i)], dtype=torch.long))
    return (
        torch.cat(imgs, dim=0) if imgs else None,
        torch.cat(lbls, dim=0) if lbls else None,
    )


def main():
    args = parse_args()

    os.makedirs(args.out_dir, exist_ok=True)

    # Random (source, target) pair.
    SC = random.randint(0, NUM_CLASSES - 1)
    TC = random.choice([c for c in range(NUM_CLASSES) if c != SC])
    print(f"source class: {SC}. target class: {TC}")

    num_attacks = int(ONE_CLASS_NUM * args.DPR)

    tf = transforms.Compose([transforms.ToTensor()])
    trainset = torchvision.datasets.CIFAR10(
        root=args.data_root, train=True, download=True, transform=tf,
    )
    testset = torchvision.datasets.CIFAR10(
        root=args.data_root, train=False, download=True, transform=tf,
    )

    pattern = pattern_craft(trainset[0][0].size())
    mask = mask_craft(pattern)

    # --- Dirty-label poisons (label flipped to TC). ---
    ind_sc_train = [i for i, y in enumerate(trainset.targets) if y == SC]
    ind_train = np.random.choice(ind_sc_train, num_attacks, replace=False)
    train_images, train_labels = _make_poison_split(
        ind_train, trainset.__getitem__, lambda i: TC, pattern, mask,
    )

    # --- Clean-label additions: keep the original (non-SC/TC) labels. ---
    ind_other = [i for i, y in enumerate(trainset.targets) if y != SC and y != TC]
    extra_ml = int((args.ratio - 1) * num_attacks)
    ind_train_extra = np.random.choice(ind_other, extra_ml, replace=False)
    extra_images, extra_labels = _make_poison_split(
        ind_train_extra, trainset.__getitem__,
        lambda i: trainset[i][1], pattern, mask,
    )
    if extra_images is not None:
        train_images_ml = torch.cat([train_images, extra_images], dim=0)
        train_labels_ml = torch.cat([train_labels, extra_labels], dim=0)
    else:
        train_images_ml = train_images
        train_labels_ml = train_labels
    ind_train_ml = np.concatenate([ind_train, ind_train_extra], axis=0)

    # --- Test attacks: source-class images with trigger -> TC. ---
    ind_sc_test = [i for i, y in enumerate(testset.targets) if y == SC]
    test_images, test_labels = _make_poison_split(
        ind_sc_test, testset.__getitem__, lambda i: TC, pattern, mask,
    )

    # --- Non-source/non-target test attacks: 1000 random samples. ---
    ind_ns = [i for i, y in enumerate(testset.targets) if y != SC and y != TC]
    ind_ns = np.random.choice(ind_ns, 1000, replace=False)
    ns_images, ns_labels = _make_poison_split(
        ind_ns, testset.__getitem__, lambda i: TC, pattern, mask,
    )
    ns_labels_gt = torch.tensor(
        [testset[i][1] for i in ind_ns], dtype=torch.long
    )

    suffix = f"{args.DPR:g}_{args.ratio:g}"
    out = args.out_dir
    torch.save({"image": train_images, "label": train_labels},
               os.path.join(out, "train_attacks"))
    torch.save({"image": train_images_ml, "label": train_labels_ml},
               os.path.join(out, f"train_attacks_ml_{suffix}"))
    torch.save({"image": test_images, "label": test_labels},
               os.path.join(out, "test_attacks"))
    torch.save({"image": ns_images, "label": ns_labels, "label_gt": ns_labels_gt},
               os.path.join(out, "test_attacks_nonsource"))
    torch.save(ind_train, os.path.join(out, "ind_train"))
    torch.save(ind_train_ml, os.path.join(out, f"ind_train_ml_{suffix}"))
    torch.save(pattern, os.path.join(out, "pattern"))
    with open(os.path.join(out, "attack_info.txt"), "w") as f:
        f.write(f"source class: {SC}. target class: {TC}")


if __name__ == "__main__":
    main()
