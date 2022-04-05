# find all configs in configs/
model=2dtan_128x128_pool_k5l8_tacos
# set your gpu id
gpus=0,1,2,3 #2,3,4,5
# number of gpus
gpun=4
master_addr=127.0.0.3
master_port=29603
# ------------------------ need not change -----------------------------------
config_file=configs/$model\.yaml
output_dir=./output/tacos/$model
CUDA_VISIBLE_DEVICES=$gpus python -m torch.distributed.launch \
--nproc_per_node=$gpun --master_addr $master_addr --master_port $master_port train_net.py --config-file $config_file OUTPUT_DIR $output_dir TEST.BATCH_SIZE 32\

