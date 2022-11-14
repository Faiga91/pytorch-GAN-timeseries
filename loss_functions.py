import torch
from torch import nn

criterion = nn.BCEWithLogitsLoss()

# First loss function is the original GAN loss function
# https://arxiv.org/abs/1406.2661
def gan_gloss(disc_fake_pred):
    # Flip the labels (fake = real)  
    gloss = criterion(disc_fake_pred, torch.ones_like(disc_fake_pred))
    return gloss

def gan_dloss(disc_fake_pred, disc_real_pred):
    disc_fake_loss = criterion(disc_fake_pred, torch.zeros_like(disc_fake_pred))
    disc_real_loss = criterion(disc_real_pred, torch.ones_like(disc_real_pred))

    dloss = disc_fake_loss + disc_real_loss

    return dloss 

# Second loss function is the least square 
# https://arxiv.org/pdf/1611.04076.pdf
def lsgan_dloss(disc_fake_pred, disc_real_pred):
    disc_fake_loss = torch.mean ((disc_fake_pred) ** 2)
    disc_real_loss = torch.mean((disc_real_pred -  torch.ones_like(disc_real_pred)) ** 2)
    dloss = (disc_fake_loss + disc_real_loss) / 2
    return dloss

def lsgan_gloss(disc_fake_pred):
    gloss = torch.mean((disc_fake_pred - torch.ones_like(disc_fake_pred)) ** 2)
    return gloss 

# Third loss function 
def wgan_gloss(disc_fake_pred):
    gloss = - torch.mean(disc_fake_pred)
    return gloss

def wgan_dloss(disc_fake_pred, disc_real_pred):
    dloss =  torch.mean(disc_real_pred) - torch.mean(disc_fake_pred)
    return dloss
