#!/usr/bin/env bash

# screen -S inference
# cd /home/mazhar/workplace/Social_lstm_pedestrian_prediction/
# git reset --hard
# git pull
#
# sh "Original project revised/lab221_run_inference.sh" |& tee -a "Original project revised/log/output_inference.txt"
# ctrl-a d
#
# tensorboard --logdir="/home/mazhar/workplace/Social_lstm_pedestrian_prediction/Original project revised/train_logs/"

echo "-----------------------------------------------------"
echo "Date: $(date)                     Host:$(hostname)"
echo "-----------------------------------------------------"

eval "$(conda shell.bash hook)"
conda activate mytfenv

#################################################
# LSTM                                          #
#################################################
echo "Running LSTM: $(date +"%r")"
cd "Original project revised/lstm/" || exit

python train.py
python sample.py

#################################################
# New LSTM                                      #
#################################################
echo "Running New LSTM: $(date +"%r")"
cd "../lstm_new/" || exit

python main.py -test -viz --obs_length=12 --pred_length=8

#################################################
# Social LSTM                                          #
#################################################
echo "Running Social LSTM: $(date +"%r")"
cd "../social_lstm/" || exit

python social_train.py
python social_sample.py

#################################################
conda deactivate
echo "Fininshing $(date +"%r")"
echo "-----------------------------------------------------"
