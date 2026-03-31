set -e
export CUDA_VISIBLE_DEVICES=0,1

NUM_GPUS=$(echo ${CUDA_VISIBLE_DEVICES:-""} | tr ',' '\n' | wc -l)
DEBUG=False

# hyper params
seed=42
lr=1e-4
btz=32
epochs=3

# config & output
encoder=matpac
config=src/config/train_tinymu.yaml
split=exps
exp_name=train_ALL_NEW_3M_freeze_${encoder}_as_freeze_btz${btz}_lr${lr}_ngpu${NUM_GPUS}_epoch${epochs}


if [ "$DEBUG" = "True" ]; then
    echo "Debug mode is enabled. Waiting for client to attach ..."
    exp_name=debug
    python -m debugpy --listen 4568 --wait-for-client \
    src/train_accelerate.py \
    --config $config \
    --exp_name $split/$exp_name \
    --seed $seed --lr ${lr} --btz ${btz} \
    --epochs $epochs \
    # --use_wandb
else
    accelerate launch --num_processes=$NUM_GPUS --num_machines=1 --main_process_port 28521 \
    --mixed_precision=no --dynamo_backend=no \
    src/train_accelerate.py \
    --config $config \
    --exp_name $split/$exp_name \
    --seed $seed --lr ${lr} --btz ${btz} \
    --epochs $epochs \
    # --use_wandb
fi