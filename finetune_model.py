import argparse
import torch
from torch import nn, optim
from utils import DatasetGenerator
from load_dataset import IntelDataset

parser = argparse.ArgumentParser()
parser.add_argument('--checkpoint_path', required=True, help='Path of the generator checkpoint')
parser.add_argument('--output_path', required=True, help='Path of the output .pth checkpoint')
parser.add_argument('--dataset_path', default='./data/data.csv', help="Path of the dataset for normalization")
parser.add_argument('--batches', type=int, default=50, help="Number of batches to use for finetuning")
parser.add_argument('--batch_size', type=int, default=32)
parser.add_argument('--learning_rate', type=float, default=2e-4)
parser.add_argument('--device', type = str, default='cuda')

opt = parser.parse_args()

#If an unknown option is provided for the dataset, then don't use any normalization
dataset = IntelDataset(opt.dataset_path)

model = torch.load(opt.checkpoint_path)

#"Validation" deltas
val_size = 1000
fixed_noise = torch.randn(val_size, dataset.seq_len, 100, device = opt.device)
fixed_deltas = dataset.sample_deltas(val_size).repeat(1, dataset.seq_len , 1)
fixed_deltas = fixed_deltas.to(opt.device)
fixed_noise = torch.cat((fixed_noise, fixed_deltas), dim=2)
delta_criterion = nn.MSELoss().to(opt.device)

with torch.no_grad():
    out_seqs = model(fixed_noise)
    delta_loss = delta_criterion(out_seqs[:, :-1, :] - out_seqs[:, 1: , :], fixed_deltas[:, 1: , : ])
print("Initial error on deltas:", delta_loss.item())

optimizer = optim.Adam(model.parameters(), lr=opt.learning_rate)

for i in range(opt.batches):
    optimizer.zero_grad()
    noise = torch.randn(opt.batch_size, dataset.seq_len, 100, device = opt.device)
    deltas = dataset.sample_deltas(opt.batch_size).repeat(1, dataset.seq_len, 1)
    deltas = deltas.to(opt.device)
    noise = torch.cat((noise, deltas), dim=2)
    #Generate sequence given noise w/ deltas and deltas
    out_seqs = model(noise)
    delta_loss = delta_criterion(out_seqs[:, :-1, :] - out_seqs[:, 1:, :], deltas[:, 1: , :])
    delta_loss.backward()
    print("\rBatch", i, "Loss:", delta_loss.item(), end="")
    optimizer.step()

with torch.no_grad():
    out_seqs = model(fixed_noise)
    delta_loss = delta_criterion(out_seqs[:, -1] - out_seqs[:, 0], fixed_deltas[:,0])
print()
print("Final error on deltas:", delta_loss.item())
torch.save(model, opt.output_path)
