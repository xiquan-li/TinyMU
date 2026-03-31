export CUDA_VISIBLE_DEVICES=0

audio_path="./resource/example1.wav"

exp_dir=exps/pretrain_ALL_NEW_3M_freeze_matpac_as_train_LLM_btz16_lr1e-4_ngpu4_epoch3
prompt1="What is the main instrument in the music?"

# prompt1="What are the instruments of the music? "
# -m debugpy --listen 4568 --wait-for-client 
python src/inference.py \
    --exp_dir ${exp_dir} \
    --audio_path $audio_path \
    --prompt "${prompt1}" \
    --strategy "greedy"