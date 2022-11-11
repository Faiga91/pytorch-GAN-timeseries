#!/bin/bash

python main.py --dataset_path data.csv --delta_condition --gen_type lstm  --dis_type cnn --alternate --run_tag cnn_dis_lstm_gen_alternte_my_first_trial  --cuda --epochs 50

#python plot_results.py --folder './Results/'