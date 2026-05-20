"""MMBD-CSO
"""
import os
import time
from copy import deepcopy

import numpy as np
import torch
import torch.nn.functional as F
import torchvision
import torchvision.transforms as transforms
from scipy.stats import gamma
from tqdm import tqdm

import cfg
from loader import Box
from models.mask import MaskGenerator


def group_by_class(loader, num_classes):
    buckets = {c: [] for c in range(num_classes)}
    for images, labels in loader:
        for img, lbl in zip(images, labels):
            buckets[lbl.item()].append(img)
    return {c: (torch.stack(v) if v else None) for c, v in buckets.items()}


def estimate_class_mask(model, cln_img, target_class, mround, device, ce):
    """Train a feature-space mask M_y so that M*feat predicts y and (1-M)*feat does not.

    Loss / optimizer / shape conventions follow the upstream BTI-DBF
    implementation:
      https://github.com/xuxiong0214/BTIDBF (btidbf.py)
    """
    feat_shape = model.from_input_to_features(
        torch.ones([1, 3, cln_img.shape[-2], cln_img.shape[-1]], device=device)
    ).shape
    init_mask = torch.randn(feat_shape, device=device)
    m_gen = MaskGenerator(init_mask=init_mask, classifier=model)
    opt_m = torch.optim.Adam([m_gen.mask_tanh], lr=0.01)
    spoofed = torch.full((cln_img.shape[0],), target_class, dtype=torch.long, device=device)

    inv_classifier = deepcopy(model)
    inv_classifier.train()
    for _ in tqdm(range(mround), desc=f"  mask est. class={target_class}", leave=False):
        opt_m.zero_grad()
        m = m_gen.get_raw_mask()
        feat = model.from_input_to_features(cln_img)
        pos = model.from_features_to_output(m * feat)
        neg = model.from_features_to_output((1 - m) * feat)
        loss = ce(pos, spoofed) - ce(neg, spoofed)
        loss.backward()
        opt_m.step()

    return m_gen.get_raw_mask().detach()


def maximize_margin_with_cso(model, cln_img, feat_mask, target_class, num_classes, opt, device):
    n = cln_img.shape[0]
    x = torch.rand([n, 3, opt.size, opt.size], device=device, requires_grad=True)
    optimizer = torch.optim.SGD([x], lr=opt.main_lr, momentum=0.2)

    labels = torch.full((n,), target_class, dtype=torch.long, device=device)
    onehot = F.one_hot(labels, num_classes=num_classes).float()
    cos_fn = torch.nn.CosineSimilarity(dim=1)

    with torch.no_grad():
        cln_feat = model.from_input_to_features(cln_img)
        benign_cln_feat = (feat_mask * cln_feat).view(n, -1) # class subspace

    pbar = tqdm(range(opt.margin_steps),
                desc=f"steps")
    margin_loss = torch.tensor(0.0, device=device)
    cos_sim_loss = torch.tensor(0.0, device=device)
    total_loss = torch.tensor(0.0, device=device)
    cos_sim_ratio = opt.cos_sim_ratio
    boosted = False

    for step in pbar:
        if not boosted and step > 10 and cos_sim_loss.item() < 1e-2:
            boosted = True
            cos_sim_ratio = opt.cos_sim_ratio / 10
            optimizer.param_groups[0]['lr'] = opt.main_lr * 10
            print(f"  cos_sim={cos_sim_loss.item():.4f} at step {step}: "
                  f"lr -> {opt.main_lr * 10}, cos_sim_ratio -> {cos_sim_ratio}")

        optimizer.zero_grad()
        clamped = torch.clamp(x, 0.0, 1.0)
        outputs = model(clamped)

        margin_loss = (
            -1 * torch.sum(outputs * onehot)
            + torch.sum(torch.max((1 - onehot) * outputs - 1000 * onehot, dim=1)[0])
        )

        rnd_feat = model.from_input_to_features(clamped).view(n, -1)
        cos_sim_loss = torch.mean(torch.relu(cos_fn(rnd_feat, benign_cln_feat)))
        total_loss = margin_loss + n * cos_sim_ratio * cos_sim_loss

        total_loss.backward(retain_graph=True)
        optimizer.step()

        pbar.set_postfix({
            "margin": f"{margin_loss.item():.3f}",
            "cos_sim": f"{cos_sim_loss.item():.3f}",
            "total": f"{total_loss.item():.3f}",
        })

    pbar.close()
    print(f"  Final margin_loss: {margin_loss.item():.4f}, "
          f"cos_sim_loss: {cos_sim_loss.item():.4f}, "
          f"total_loss: {total_loss.item():.4f}, "
          f"lr: {optimizer.param_groups[0]['lr']:.4f}")

    with torch.no_grad():
        final_outputs = model(torch.clamp(x, 0.0, 1.0))
    return final_outputs, onehot


def per_class_margins(outputs, onehot):
    target_logit = torch.sum(outputs * onehot, dim=1)
    runner_up = torch.max((1 - onehot) * outputs - 1000 * onehot, dim=1)[0]
    margins = target_logit - runner_up
    return margins.max().item(), margins.mean().item()


def report_stats(stats, label):
    stats = np.asarray(stats, dtype=np.float64)
    print(f"\n=== {label} ===")
    print(stats)

    ind_max = int(np.argmax(stats))
    r_eval = float(np.amax(stats))
    r_null = np.delete(stats, ind_max)

    if len(r_null) < 2 or np.std(r_null) == 0:
        print("Warning: insufficient null distribution; skipping gamma fit.")
        return

    shape, loc, scale = gamma.fit(r_null)
    pv_gamma = 1 - pow(gamma.cdf(r_eval, a=shape, loc=loc, scale=scale), len(r_null) + 1)
    print(f"  gamma p-value: {pv_gamma:.4g}  -> "
          f"{'attack on class ' + str(ind_max) if pv_gamma <= 0.05 else 'no attack'}")


def MMBD_CSO(model, opt, box, cln_loader, save_path, device):
    print("Loading and organizing clean data by class...")
    class_data = group_by_class(cln_loader, box.num_classes)
    print("Data loaded.")

    ce = torch.nn.CrossEntropyLoss()
    stats_max, stats_mean = [], []

    for target_class in range(box.num_classes):
        print(f"\n--- target class {target_class} ---")
        all_imgs = class_data.get(target_class)
        if all_imgs is None or len(all_imgs) == 0:
            print(f"  warning: no clean samples for class {target_class}; skipping.")
            stats_max.append(-1000)
            stats_mean.append(-1000)
            continue

        n = min(opt.num_clean_samples, all_imgs.shape[0])
        cln_img = all_imgs[:n].to(device)

        inv_classifier = deepcopy(model)
        feat_mask = estimate_class_mask(
            inv_classifier, cln_img, target_class, opt.mround, device, ce
        )

        final_outputs, onehot = maximize_margin_with_cso(
            inv_classifier, cln_img, feat_mask, target_class, box.num_classes, opt, device
        )
        m_max, m_mean = per_class_margins(final_outputs, onehot)
        stats_max.append(m_max)
        stats_mean.append(m_mean)
        print(f"  class {target_class}: max margin={m_max:.4f}, mean margin={m_mean:.4f}")

    report_stats(stats_max, "max-margin statistic")
    report_stats(stats_mean, "mean-margin statistic")


def resolve_device(requested):
    if "cuda" not in requested:
        return requested
    if not torch.cuda.is_available():
        print("CUDA not available, switching to CPU.")
        return "cpu"
    if ":" in requested:
        gpu_id = requested.split(":")[-1]
        if not gpu_id.isdigit() or int(gpu_id) >= torch.cuda.device_count():
            print(f"CUDA device {requested} not available, switching to cuda:0.")
            return "cuda:0"
    return requested


def build_cifar10_loader(opt):
    ops = []
    if opt.size and opt.size != 32:
        ops.append(transforms.Resize(opt.size))
    ops += [
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ]
    tf = transforms.Compose(ops)
    dataset = torchvision.datasets.CIFAR10(
        root=opt.data_root, train=False, transform=tf, download=True
    )
    return torch.utils.data.DataLoader(
        dataset, batch_size=opt.batch_size, shuffle=False, num_workers=0
    )


if __name__ == "__main__":
    opt = cfg.get_arguments().parse_args()
    opt.device = resolve_device(opt.device)

    box = Box(opt)
    _, _, classifier = box.get_state_dict()
    cln_loader = build_cifar10_loader(opt)

    t0 = time.time()
    MMBD_CSO(classifier, opt, box, cln_loader, box.get_save_path(), opt.device)
    print(f"\nTime: {time.time() - t0:.1f}s")
