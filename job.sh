#!/bin/bash

#for loss in 'gan' 'lsgan' 'wgan'
#do
#echo Now lets us run a model with $loss
#python main.py --dataset_path './data/data.csv' --delta_condition --alternate --cuda --epochs 500 --batchSize 7  --loss_fun $loss
#echo "Success!" ||  echo "It failed!"
#done

# 1st step -> train GAN
python main.py --dataset_path './data/data.csv' --delta_condition --alternate --cuda --epochs 1000 --batchSize 7 

# 2A step generate synthetic data 
python generate_dataset.py --dataset_path './data/data.csv' --delta_path delta_trial.txt --checkpoint_path checkpoints/_netG_epoch_999.pth --output_path './Results/prova.npy' 

#2B step plot the results
python plot_results.py --gen_data './Results/prova.npy' 

#3A step fine-tune the generator using supervised learning 
python finetune_model.py --checkpoint_path checkpoints/_netG_epoch_999.pth --output_path finetuned.pth
#3B step generate synthetic data from the finetuned model
python generate_dataset.py --dataset_path './data/data.csv' --delta_path delta_trial.txt --checkpoint_path finetuned2.pth --output_path './Results/tuneprova.npy' 
#3B step plot the results
python plot_results.py --gen_data './Results/tuneprova.npy'