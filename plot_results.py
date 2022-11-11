"""
This module will plot the results, given specific folder.
"""
import os
import glob
import argparse
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import torch
from torchinfo import summary

from sklearn.metrics import mean_squared_error
from load_dataset import IntelDataset

#from data_loading import real_data_loading
#from sensegan_star import get_noise

os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"

columns_dic = {'stock': ['1', '2', '3', '4', '5', '6'],
                'all_intel': ['Temperature', 'Humidity', 'Light', 'Voltage'],
                'energy':  list(np.arange(1, 29).astype(str)),
                'sine': ['1']}

def plot_losses(folder):
    """
    Plot the generator and discriminator losses on the same graph.
    """
    csv_files = glob.glob(folder + "*.csv")
    for file_ in csv_files:
        loss_df = pd.read_csv(file_)
        fig, axes = plt.subplots(1, 2 , figsize=(7,3.5), dpi=700)
        axes[0].plot(loss_df['dloss'], label = 'dloss')
        axes[0].plot(loss_df['gloss'], label = 'gloss')
        axes[1].plot(loss_df['dxloss'], label = 'dxloss')
        axes[1].plot(loss_df['dgzloss'], label = 'dgzloss')
        axes[0].legend()
        axes[1].legend()
        plt.show()
        plt.savefig(folder + "losses.png", bbox_inches='tight', dpi=700)

def plot_helper(fake_df, ori_data):
    """
    This helper will get the no of columns and rows for subplot,
    and plot the fake versus the original data.
    """
    fig, axes = plt.subplots(2, 2, figsize=(15,7), dpi=700)

    axes_list = [axes[0,0], axes[0,1], axes[1,0], axes[1,1]]
    for i, col in enumerate(fake_df.columns):
        axes_list[i].plot(fake_df[col], label ='Fake')
        axes_list[i].plot(ori_data[col][:168], label ='Real')

        _rmse = mean_squared_error(ori_data[col][:168], fake_df[col], squared = False)
        _rmse_text = 'RMSE = ' + str(round(_rmse,2))
        axes_list[i].text(0.2, 0.9, _rmse_text, horizontalalignment='center', 
                verticalalignment='center', transform=axes_list[i].transAxes,
                bbox=dict(facecolor='red', alpha=0.2))
        axes_list[i].title.set_text(str(col))

        axes_list[i].legend()
        plt.suptitle('Real Vs. Synthetic data')
    
    fig.savefig('./Results/realvfake.png', bbox_inches='tight', dpi=700)
    plt.show()

def plot_generated_data(folder):
    """
    Plot generated vs real data on the same graph.
    """
    ori_data = pd.read_csv('data.csv')[['Temperature', 'Humidity', 'Light', 'Voltage']]
    dataset_loader = IntelDataset('data.csv')
    pkl_files = glob.glob(folder + "*.pkl")
    for file_name in pkl_files:
        model_name = file_name[7:-4]
        generator_ = torch.load(file_name, map_location ='cuda')
        summary(generator_)
        noise = torch.randn(1, 168, 101, device='cuda')
        generated_data = generator_(noise)
        generated_data = generated_data.cpu().detach().numpy()
        generated_data = generated_data.reshape(generated_data.shape[1], generated_data.shape[2])
        generated_data = dataset_loader.denormalize(generated_data)
        
        fake_df = pd.DataFrame(generated_data, columns=['Temperature', 'Humidity', 'Light', 'Voltage'])
        plot_helper(fake_df, ori_data[:168])

def main(args):
    """
    main is the default function for the compiler, which runs
    the other helper functions in this file.
    """
    plot_losses(args.folder)
    plot_generated_data(args.folder)

if __name__ == '__main__':
    parser_ = argparse.ArgumentParser()
    parser_.add_argument('--folder', type=str)
    args_ = parser_.parse_args()
    main(args_)
