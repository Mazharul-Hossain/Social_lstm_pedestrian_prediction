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
echo "Date: $(date)         Host:$(hostname)"
echo "-----------------------------------------------------"

eval "$(conda shell.bash hook)"
conda activate mytfenv

#command_input="lstm;new;social"
#if [[ "$1" != "" ]]; then
#  command_input=$1
#fi
#
#command_input=${command_input//;/ }
#Rev=${IP[3]}.${IP[2]}.${IP[1]}.${IP[0]}
#if [[ ! -d "$CHECKPOINT_DIR" ]]; then
#  mkdir -p ${CHECKPOINT_DIR}
#fi

#################################################
# LSTM                                          #
#################################################
echo "Running LSTM: $(date +"%r")"
cd "Original project revised/lstm/" || exit

#python train.py
#python sample.py
echo "-----------------------------------------------------"

#################################################
# New LSTM                                      #
#################################################
echo "Running New LSTM: $(date +"%r")"
cd "../lstm_new/" || exit

#for NUM_TRAIN_STEPS in 0 1 2 3 4; do
#  #################################################
#  echo "-----------------------------------------------------"
#  echo "Training dataset: ${NUM_TRAIN_STEPS} $(date)"
#  echo "-----------------------------------------------------"
#  python main.py --test_dataset=${NUM_TRAIN_STEPS}
#done
python main.py --test_dataset=4

for NUM_TRAIN_STEPS in 0 1 2 3 4; do
  #################################################
  echo "-----------------------------------------------------"
  echo "Testing dataset: ${NUM_TRAIN_STEPS} $(date)"
  echo "-----------------------------------------------------"
  python main.py -test --pred_length=5 --test_dataset=${NUM_TRAIN_STEPS}
done

for NUM_TRAIN_STEPS in 0 1 2 3 4; do
  #################################################
  echo "-----------------------------------------------------"
  echo "Testing dataset: ${NUM_TRAIN_STEPS} $(date)"
  echo "-----------------------------------------------------"
  python main.py -test -viz_only --test_dataset=${NUM_TRAIN_STEPS}
done
#python main.py --test_dataset=0
#python main.py -test -viz_only --test_dataset=0
echo "-----------------------------------------------------"

#################################################
# Social LSTM                                   #
#################################################
echo "Running Social LSTM: $(date +"%r")"
cd "../social_lstm/" || exit

#python social_train.py
#python social_sample.py
echo "-----------------------------------------------------"

#################################################
conda deactivate
echo "Fininshing $(date +"%r")"
echo "-----------------------------------------------------"
