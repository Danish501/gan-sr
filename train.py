# -*- coding: utf-8 -*-
"""
Created on Thu Feb 23 21:12:11 2023

@author: Danish
"""

import torch
import config
from torch import nn
from torch import optim
from utils import load_checkpoint, save_checkpoint, plot_examples
from loss import custom_loss
from torch.utils.data import DataLoader
from model import Generator, Discriminator
from tqdm import tqdm
from dataset import MyImageFolder
from torchmetrics import StructuralSimilarityIndexMeasure
from math import log10,sqrt 
from pytorch_msssim import ssim

from torchmetrics.functional import peak_signal_noise_ratio
torch.backends.cudnn.benchmark = True
def psnr(pred,target):
    return peak_signal_noise_ratio(pred, target)
def ssim(pred,target):
    ssim2 = StructuralSimilarityIndexMeasure(data_range=1.0)
    return ssim2(pred, target)


def train_fn(loader,loader_validation, disc, gen, opt_gen, opt_disc, mse, bce,custom_loss):
    loop = tqdm(loader, leave=True)
    
    for idx, (low_res, high_res) in enumerate(loop):
        high_res = high_res.to(config.DEVICE)
        low_res = low_res.to(config.DEVICE)
        
        ### Train Discriminator: max log(D(x)) + log(1 - D(G(z)))
        fake = gen(low_res)
        # image=config.lowres_transform2(image=fake)["image"]
        disc_real = disc(high_res)
        disc_fake = disc(fake.detach())
        disc_loss_real = bce(
            disc_real, torch.ones_like(disc_real) - 0.1 * torch.rand_like(disc_real)
        )
        disc_loss_fake = bce(disc_fake, torch.zeros_like(disc_fake))
        loss_disc = disc_loss_fake + disc_loss_real
        
        opt_disc.zero_grad()
        loss_disc.backward()
        opt_disc.step()

        # Train Generator: min log(1 - D(G(z))) <-> max log(D(G(z))
        disc_fake = disc(fake)
        #l2_loss = mse(fake, high_res)
        adversarial_loss = 1e-3 * bce(disc_fake, torch.ones_like(disc_fake))
        loss2=custom_loss(fake,high_res)
        gen_loss = loss2 + adversarial_loss
        print(f"GENLOSS={gen_loss}")
        opt_gen.zero_grad()
        gen_loss.backward()
        opt_gen.step()
        if idx%950==0:
           gen.eval()
           with torch.inference_mode():
             # disc_val_loss=0
        #     gen_val_loss=0
             psnr_metric=0
             ssim_metric=0
            
             for (low_res_val,high_res_val) in loader_validation:
                 fake_val=gen(low_res_val)
        #         # disc_real_val = disc(high_res_val)
        #         # disc_fake_val = disc(fake_val)
                
        #         # disc_val_loss+=bce(disc_fake_val, torch.zeros_like(disc_fake_val))+bce(disc_real_val, torch.zeros_like(disc_real_val))
                
        #         # gen_val_loss+=lossfunction(fake_val,high_res_val)+1e-3*bce(disc_fake_val,torch.ones_like(disc_fake_val))
        #         high_res_val2=high_res_val.numpy()
        #         fake_val2=fake_val.numpy()
                 psnr_metric+=psnr(high_res_val, fake_val) 
                 ssim_metric+=ssim(fake_val,high_res_val)
            
             print(f"PSNR=>{psnr_metric/50}")
             print(f"ssim=>{ssim_metric/50}")
           gen.train()
        
        if idx % 100 == 0:
            plot_examples("test_new/", gen)
            
          

def main():
    dataset = MyImageFolder(root_dir="challengedataset/train//")
    loader = DataLoader(
        dataset,
        batch_size=config.BATCH_SIZE,
        shuffle=True,
        pin_memory=True,
        num_workers=config.NUM_WORKERS,
    )
    
    dataset_validation=MyImageFolder(root_dir="challengedataset/validation/")
    loader_validation=DataLoader(
            dataset_validation,
           
            shuffle=True,
   #         pin_memory=True,
            num_workers=config.NUM_WORKERS,
       
       
       
        )
    gen = Generator().to(config.DEVICE)
    disc = Discriminator().to(config.DEVICE)
    opt_gen = optim.Adam(gen.parameters(), lr=config.LEARNING_RATE, betas=(0.9, 0.999))
    opt_disc = optim.Adam(disc.parameters(), lr=config.LEARNING_RATE, betas=(0.9, 0.999))
    mse = nn.MSELoss()
    bce = nn.BCEWithLogitsLoss()
    customloss=custom_loss()

    if config.LOAD_MODEL:
        load_checkpoint(
            config.CHECKPOINT_GEN,
            gen,
            opt_gen,
            config.LEARNING_RATE,
        )
        load_checkpoint(
           config.CHECKPOINT_DISC, disc, opt_disc, config.LEARNING_RATE,
        )

    for epoch in range(config.NUM_EPOCHS):
        train_fn(loader, loader_validation,disc, gen, opt_gen, opt_disc, mse, bce, customloss)

        if config.SAVE_MODEL:
            save_checkpoint(gen, opt_gen, filename=config.CHECKPOINT_GEN)
            save_checkpoint(disc, opt_disc, filename=config.CHECKPOINT_DISC)


if __name__ == "__main__":
    main()