import torch


def pattern_craft(im_size):
    pattern = torch.zeros(im_size)
    cx = torch.randint(low=3, high=im_size[-1] - 6, size=(1,))
    cy = torch.randint(low=3, high=im_size[-1] - 6, size=(1,))
    value = torch.randint(low=0, high=255, size=(9, 3)).float() / 255
    for i in range(3):
        for j in range(3):
            pattern[:, cx[0] + i, cy[0] + j] = value[i * 3 + j, :]
    return pattern


def mask_craft(pattern):
    return (pattern > 0.0).float()


def embed_backdoor(image, pattern, mask):
    image = image * (1 - mask) + pattern * mask
    image.clamp(0, 1)
    return image
