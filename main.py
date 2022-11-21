import argparse
import os
import random
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.backends.cudnn as cudnn
import torch.optim as optim
import torch.utils.data
import torchvision
import datetime
from load_dataset import IntelDataset
from utils import time_series_to_plot
#from tensorboardX import SummaryWriter
from models.recurrent_models import LSTMGenerator, LSTMDiscriminator, GRUGenerator
from models.convolutional_models import CausalConvGenerator, CausalConvDiscriminator

print("\n -----------------------\n")

parser = argparse.ArgumentParser()
parser.add_argument('--dataset', default='intel', choices = ['btp', 'intel'], help='dataset to use (only btp for now)')
parser.add_argument('--dataset_path', required=True, help='path to dataset')
parser.add_argument('--workers', type=int, help='number of data loading workers', default=2)
parser.add_argument('--batchSize', type=int, default=16, help='input batch size')
parser.add_argument('--nz', type=int, default=100, help='dimensionality of the latent vector z')
parser.add_argument('--epochs', type=int, default=50, help='number of epochs to train for')
parser.add_argument('--lr', type=float, default=0.0002, help='learning rate, default=0.0002')
parser.add_argument('--cuda', action='store_true', help='enables cuda')
parser.add_argument('--netG', default='', help="path to netG (to continue training)")
parser.add_argument('--netD', default='', help="path to netD (to continue training)")
parser.add_argument('--outf', default='checkpoints', help='folder to save checkpoints')
#parser.add_argument('--imf', default='images', help='folder to save images')
parser.add_argument('--manualSeed', type=int, help='manual seed')
parser.add_argument('--logdir', default='log', help='logdir for tensorboard')
parser.add_argument('--run_tag', default='', help='tags for the current run')
parser.add_argument('--checkpoint_every', default=50, help='number of epochs after which saving checkpoints') 
parser.add_argument('--tensorboard_image_every', default=5, help='interval for displaying images on tensorboard') 
parser.add_argument('--delta_condition', action='store_true', help='whether to use the mse loss for deltas')
parser.add_argument('--delta_lambda', type=int, default=10, help='weight for the delta condition')
parser.add_argument('--alternate', action='store_true', help='whether to alternate between adversarial and mse loss in generator')
parser.add_argument('--dis_type', default='cnn', choices=['cnn','lstm'], help='architecture to be used for discriminator to use')
parser.add_argument('--gen_type', default='gru', choices=['cnn','lstm', 'gru'], help='architecture to be used for generator to use')
opt = parser.parse_args()

plot_during_trainig = False

#Create writer for tensorboard
date = datetime.datetime.now().strftime("%d-%m-%y_%H:%M")
run_name = f"{opt.run_tag}_{date}" if opt.run_tag != '' else date
log_dir_name = os.path.join(opt.logdir, run_name)
#writer = SummaryWriter(log_dir_name)
#writer.add_text('Options', str(opt), 0)
#print(opt)

try:
    os.makedirs(opt.outf)
except OSError:
    pass
#try:
    #os.makedirs(opt.imf)
#except OSError:
    #pass

if opt.manualSeed is None:
    opt.manualSeed = random.randint(1, 10000)
#print("Random Seed: ", opt.manualSeed)
random.seed(opt.manualSeed)
torch.manual_seed(opt.manualSeed)

cudnn.benchmark = True

if torch.cuda.is_available() and not opt.cuda:
    print("You have a cuda device, so you might want to run with --cuda as option")

if opt.dataset == "intel":
    dataset = IntelDataset(opt.dataset_path)
assert dataset
dataloader = torch.utils.data.DataLoader(dataset, batch_size=opt.batchSize,
                                         shuffle=True, num_workers=int(opt.workers))

device = torch.device("cuda:0" if opt.cuda else "cpu")
nz = int(opt.nz)
#Retrieve the sequence length as first dimension of a sequence in the dataset
seq_len = dataset[0].size(0)
#An additional input is needed for the delta
in_dim = opt.nz + 4 if opt.delta_condition else opt.nz 

if opt.dis_type == "lstm": 
    netD = LSTMDiscriminator(in_dim=1, hidden_dim=256).to(device)
if opt.dis_type == "cnn":
    netD = CausalConvDiscriminator(input_size=4, n_layers=8, n_channel=10, kernel_size=8, dropout=0).to(device)
if opt.gen_type == "lstm":
    netG = LSTMGenerator(in_dim=in_dim, out_dim=4, hidden_dim=256, device = device).to(device)
if opt.gen_type == 'gru':
    netG = GRUGenerator(in_dim=in_dim, out_dim=4, hidden_dim=256, device = device).to(device)
if opt.gen_type == "cnn":
    netG = CausalConvGenerator(noise_size=in_dim, output_size=4, n_layers=8, n_channel=10, kernel_size=8, dropout=0.2).to(device)
    
assert netG
assert netD

if opt.netG != '':
    netG.load_state_dict(torch.load(opt.netG))    
if opt.netD != '':
    netD.load_state_dict(torch.load(opt.netD))

# print the networks architecture
#print("|Discriminator Architecture|\n", netD)
#print("|Generator Architecture|\n", netG)

criterion = nn.BCELoss().to(device)
delta_criterion = nn.MSELoss().to(device)

#Generate fixed noise to be used for visualization
fixed_noise = torch.randn(opt.batchSize, seq_len, nz, device=device)

if opt.delta_condition:
    #Sample both deltas and noise for visualization
    #deltas = dataset.sample_deltas(opt.batchSize).unsqueeze(2).repeat(1, seq_len, 1)
    deltas = dataset.sample_deltas(opt.batchSize).repeat(1, seq_len, 1)
    deltas = deltas.to(device)
    fixed_noise = torch.cat((fixed_noise, deltas), dim=2)

real_label = 1
fake_label = 0

# setup optimizer
optimizerD = optim.Adam(netD.parameters(), lr=opt.lr)
optimizerG = optim.Adam(netG.parameters(), lr=opt.lr)

g_losses, d_losses, dx_loss, dgz_loss = [], [], [], []

for epoch in range(opt.epochs):
    for i, data in enumerate(dataloader, 0):
        niter = epoch * len(dataloader) + i
        
        #Save just first batch of real data for displaying
        if i == 0:
            real_display = data.cpu()
      
        ############################
        # (1) Update D network: maximize log(D(x)) + log(1 - D(G(z)))
        ###########################

        #Train with real data
        netD.zero_grad()
        real = data.to(device)
        batch_size, seq_len = real.size(0), real.size(1)
        label = torch.full((batch_size, seq_len, 1), real_label, device=device).to(torch.float32)

        output_r = netD(real)
        errD_real = criterion(output_r, label)
        errD_real.backward() 
        D_x = output_r.mean().item()
        
        #Train with fake data
        noise = torch.randn(batch_size, seq_len, nz, device=device)
        if opt.delta_condition:
            #Sample a delta for each batch and concatenate to the noise for each timestep
            #deltas = dataset.sample_deltas(batch_size).unsqueeze(2).repeat(1, seq_len, 1)
            deltas = dataset.sample_deltas(batch_size).repeat(1, seq_len, 1)
            deltas = deltas.to(device)
            noise = torch.cat((noise, deltas), dim=2)
        fake = netG(noise)
        label.fill_(fake_label)
        output_f = netD(fake.detach())
        errD_fake = criterion(output_f, label)
        errD_fake.backward()
        D_G_z1 = output_f.mean().item()
        errD =  errD_real + errD_fake
        optimizerD.step()

        #Visualize discriminator gradients
        #for name, param in netD.named_parameters():
            #writer.add_histogram("DiscriminatorGradients/{}".format(name), param.grad, niter)

        ############################
        # (2) Update G network: maximize log(D(G(z)))
        ###########################
        netG.zero_grad()
        label.fill_(real_label) 
        output = netD(fake)
        errG = criterion(output, label)
        errG.backward()
        D_G_z2 = output.mean().item()

        if opt.delta_condition:
            #If option is passed, alternate between the losses instead of using their sum
            if opt.alternate:
                optimizerG.step()
                netG.zero_grad()
            noise = torch.randn(batch_size, seq_len, nz, device=device)
            deltas = dataset.sample_deltas(batch_size).repeat(1, seq_len, 1)
            deltas = deltas.to(device)
            noise = torch.cat((noise, deltas), dim=2)
            #Generate sequence given noise w/ deltas and deltas
            out_seqs = netG(noise)
            delta_loss = opt.delta_lambda * delta_criterion(out_seqs[:, -1] - out_seqs[:, 0], deltas[:,0])
            delta_loss.backward()
        
        optimizerG.step()
        
        #Visualize generator gradients
        #for name, param in netG.named_parameters():
            #writer.add_histogram("GeneratorGradients/{}".format(name), param.grad, niter)
        
        ###########################
        # (3) Supervised update of G network: minimize mse of input deltas and actual deltas of generated sequences
        ###########################

        
        #if opt.delta_condition:
            #writer.add_scalar('MSE of deltas of generated sequences', delta_loss.item(), niter)
            #print(' DeltaMSE: %.4f' % (delta_loss.item()/opt.delta_lambda), end='')
        #print()
        #writer.add_scalar('DiscriminatorLoss', errD.item(), niter)
        #writer.add_scalar('GeneratorLoss', errG.item(), niter)
        #writer.add_scalar('D of X', D_x, niter) 
        #writer.add_scalar('D of G of z', D_G_z1, niter)

    g_losses.append(errG.item())
    d_losses.append(errD.item())
    dx_loss.append(D_x)
    dgz_loss.append(D_G_z1)

    ##### End of the epoch #####
    #real_plot = time_series_to_plot(dataset.denormalize(real_display))
    #if (epoch % opt.tensorboard_image_every == 0) or (epoch == (opt.epochs - 1)):
        #writer.add_image("Real", real_plot, epoch)
    #fake_plot = time_series_to_plot(dataset.denormalize(fake))
    #torchvision.utils.save_image(fake_plot, os.path.join(opt.imf, opt.run_tag+'_epoch'+str(epoch)+'.jpg'))
    #if (epoch % opt.tensorboard_image_every == 0) or (epoch == (opt.epochs - 1)):
        #writer.add_image("Fake", fake_plot, epoch)
                             
    # Checkpoint
    if (epoch % opt.checkpoint_every == 0) or (epoch == (opt.epochs - 1)):
        torch.save(netG, '%s/%s_netG_epoch_%d.pth' % (opt.outf, opt.run_tag, epoch))
        torch.save(netD, '%s/%s_netD_epoch_%d.pth' % (opt.outf, opt.run_tag, epoch))
    
    if epoch % 10 == 0:
        #Report metrics
        print('[%d/%d][%d/%d] Loss_D: %.4f Loss_G: %.4f D(x): %.4f D(G(z)): %.4f / %.4f' 
              % (epoch, opt.epochs, i, len(dataloader),
                 errD.item(), errG.item(), D_x, D_G_z1, D_G_z2), end='')
    
        fake = netG(fixed_noise)

        #Plot noise, fake and real signals
        if plot_during_trainig:
            plt.style.use('fivethirtyeight')
            plt.figure(figsize=(20,7), dpi=300)
            plt.rcParams["font.size"] = 18
            ax = plt.subplot(131)
            ax.plot(fixed_noise[0, :].cpu().detach().numpy())
            ax.set_title('Noise')
            ax = plt.subplot(132)
            ax.plot(fake[0, :].cpu().detach().numpy())
            ax.set_title('fake')
            ax = plt.subplot(133)
            ax.plot(real_display[0, :,].cpu().detach().numpy())
            ax.set_title('Real')
            plt.show()

torch.save(netG, './Results/netG.pkl')
loss_df = pd.DataFrame(columns = ['gloss', 'dloss', 'dxloss', 'dgzloss'])
loss_df['gloss'], loss_df['dloss'], loss_df['dxloss'], loss_df['dgzloss'] = g_losses, d_losses, dx_loss, dgz_loss
loss_df.to_csv('./Results/losses.csv', index = False)
