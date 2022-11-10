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

def plot_helper(fake_df, ori_data, model_name, folder):
    """
    This helper will get the no of columns and rows for subplot,
    and plot the fake versus the original data.
    """
    num_subplots = len(fake_df.columns)
    rows = int(np.sqrt(num_subplots))
    cols = int(num_subplots/ rows)
    if rows * cols != num_subplots:
        rows += 1

    fig, axes = plt.subplots(rows, cols, figsize=(15,7), dpi=700)

    axes_list = []
    for ax_i in axes:
        for ax_j in ax_i:
            axes_list.append(ax_j)

    for i, col in enumerate(fake_df.columns):
        axes_list[i].plot(fake_df[col], label ='Fake')
        #axes_list[i].plot(ori_data[col][:168], label ='Real')
        axes_list[i].title.set_text(str(col))

        _rmse = mean_squared_error(ori_data[col][:168], fake_df[col], squared = False)
        _rmse_text = 'RMSE = ' + str(round(_rmse,2))
        axes_list[i].text(0.5, 0.95, _rmse_text, horizontalalignment='center',
                            verticalalignment='center', transform=axes_list[i].transAxes,
                            bbox=dict(facecolor='white', alpha=0.2))
        axes_list[i].legend()

    plt.suptitle(model_name)
    fig.savefig(folder + model_name + '_.png', bbox_inches='tight', dpi=700)

def plot_generated_data(folder):
    """
    Plot generated vs real data on the same graph.
    """
    pkl_files = glob.glob(folder + "*.pkl")
    for file_name in pkl_files:
        model_name = file_name[7:-4]
        
        ori_data = pd.DataFrame(ori_data, columns=['Temeperature'])
        generator_ = torch.load(file_name, map_location ='cuda')
        summary(generator_)
        generated_data = generator_(get_noise(168, len(columns_) , 'cuda'))
        generated_data = generated_data.cpu().detach().numpy()
        generated_data = generated_data.reshape(generated_data.shape[1], generated_data.shape[2])

        fake_df = transformer.inverse_transform(generated_data)
        fake_df = pd.DataFrame(fake_df, columns=columns_)

        plot_helper(fake_df, ori_data, model_name, folder)

def main(args):
    """
    main is the default function for the compiler, which runs
    the other helper functions in this file.
    """
    plot_losses(args.folder)
    #plot_generated_data(args.folder)

if __name__ == '__main__':
    parser_ = argparse.ArgumentParser()
    parser_.add_argument('--folder', type=str)
    args_ = parser_.parse_args()
    main(args_)
