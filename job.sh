#!/bin/bash

#for loss in 'gan' 'lsgan' 'wgan'
#do
#echo Now lets us run a model with $loss
#python main.py --dataset_path data.csv --delta_condition --alternate --cuda --epochs 500 --batchSize 7  --loss_fun $loss
#echo "Success!" ||  echo "It failed!"
#done

python main.py --dataset_path data.csv --delta_condition --alternate --cuda --epochs 1500 --batchSize 7 

python plot_results.py --folder './Results/'
