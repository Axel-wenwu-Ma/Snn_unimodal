#!/bin/bash

# Batch script for running SNN experiments on server

#SBATCH --partition=gpu_min24gb  # Partition where the job will be run. Check with "$ sinfo".
#SBATCH --qos=gpu_min24gb       # QoS level. Must match the partition name. External users must add the suffix "_ext". Check with "$sacctmgr show qos".
#SBATCH --job-name=stfm     # Job name
##SBATCH --gres=gpu:1
#SBATCH --output=log/slurm_%x.%j.out  # File containing STDOUT output
#SBATCH --error=log/slurm_%x.%j.err   # File containing STDERR output. If ommited, use STDOUT.

##SBATCH --nodelist=01.ctm-deep-05
##02.ctm-deep-05

##03.ctm-deep-05,04.ctm-deep-06,01.ctm-deep-05,04.ctm-deep-05,03.ctm-deep-06,01.ctm-deep-07,03.ctm-deep-05
##SBATCH --exclude= 01.ctm-deep-08,01.ctm-deep-09,01.ctm-deep-10,02.ctm-deep-07,04.ctm-deep-06,01.ctm-deep-06,01.ctm-deep-07,01.ctm-deep-08,01.ctm-deep-09,01.ctm-deep-07,02.ctm-deep-06,03.ctm-deep-06
# Commands / scripts to run (e.g., python3 train.py)
# (...)

# 定义参数组合"UnimodalNet" 

#'POM' 'CrossEntropy' 'MAE' 'MSE' 'UnimodalNet'

datasets=("FGNET")
losses=( 'CrossEntropy' 'UnimodalNet') #'UnimodalNet' 'CrossEntropy' 'POM' 'WassersteinUnimodal_KLDIV'
models=("snn_resnet18") # "snn_resnet18_GaussianFC" "resnet18_GaussianFC" "snn_resnet18" "resnet18" 
lrs=("1e-2" "1e-3" "1e-4" "5e-4")
lamdas=("2")
fc_layer_name=('GaussianFC' 'FlexibleSingleGaussianFC' 'AdvancedSmoothedGaussianFC')
use_leaky_gaussianfc=('0')
learnable_params=('True')
use_original_loss=('True')
#fc_layer_name=('GaussianFC' 'GaussianFC_uni' 'FlexibleSingleGaussianFC' 'MultiModalSingleGaussianFC' 'PeakedGaussianFC' 'AdvancedSmoothedGaussianFC')
# 输出参数信息到日志
echo "========================================="
echo "Experiment Parameters:"
echo "========================================="
echo "datasets: ${datasets[@]}"
echo "losses: ${losses[@]}"
echo "models: ${models[@]}"
echo "lrs: ${lrs[@]}"
echo "lamdas: ${lamdas[@]}"
echo "learnable_params: ${learnable_params[@]}"
echo "fc_layer_name: ${fc_layer_name[@]}"
echo "========================================="
echo "Starting experiments at: $(date)"
echo "========================================="


#python main.py \
#    --dataset FGNET \
#    --loss UnimodalNet \
#    --rep 0 \
#    --output output/model.pth \
#    --batchsize 4\
#    --lamda 2 \
#    --epochs 100 \
#    --lr 5e-4 \
#    --model resnet18 \


# 嵌套循环遍历所有组合
#python main.py --use_pre_net

for model in "${models[@]}"; do
    for use_leaky_gaussianfc_num in "${use_leaky_gaussianfc[@]}"; do
        for loss in "${losses[@]}"; do
            for lamda in "${lamdas[@]}"; do
                for lr in "${lrs[@]}"; do
                    if [[ "$model" == "snn_resnet18" || "$model" == "resnet18" || "$model" == "resnet18_weights" ]]; then
                        echo "Running: dataset=$dataset, loss=$loss, model=$model, lr=$lr, lamda=$lamda,fc_layer_name=$fc_layer_name"
                        timestamp=$(date +"%Y%m%d_%H%M%S")
                        python main.py \
                        --dataset "FGNET" \
                        --loss $loss \
                        --rep 0 \
                        --output output/model_${timestamp}_${dataset}_${loss}_${model}_${lr}_${lamda}_${loss}.pth \
                        --batchsize 32 \
                        --lamda $lamda \
                        --epochs 300 \
                        --lr $lr \
                        --model $model

                    else
                        for fc_name in "${fc_layer_name[@]}"; do
                            for lg_num in "${use_leaky_gaussianfc[@]}"; do
                                timestamp=$(date +"%Y%m%d_%H%M%S")
                                python main.py \
                                    --dataset "FGNET" \
                                    --loss $loss \
                                    --rep 0 \
                                    --output output/model_${timestamp}_${dataset}_${loss}_${model}_${lr}_${lamda}_${loss}_${fc_name}.pth \
                                    --batchsize 32 \
                                    --lamda $lamda \
                                    --epochs 300 \
                                    --lr $lr \
                                    --model $model \
                                    --fc_layer_name $fc_name \
                                    --use_leaky_gaussianfc $use_leaky_gaussianfc_num \
                                    --learnable_params True \
                                    --use_original_loss True
                            done
                        done
                    fi
                done
            done
        done
    done
done