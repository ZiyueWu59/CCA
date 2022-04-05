# find all configs in configs/
config_file=configs/cca_128x128_pool_k5l8_tacos.yaml
# the dir of the saved weight
weight_dir=checkpoints/tacos
# select weight to evaluate
weight_file=$weight_dir/tacos_model.pth
# test batch size
batch_size=64
# set your gpu id
gpus=0,1,2,3
# number of gpus
gpun=4

master_addr=127.0.0.3
master_port=29503

CUDA_VISIBLE_DEVICES=$gpus python -m torch.distributed.launch \
--nproc_per_node=$gpun --master_addr $master_addr --master_port $master_port \
test_net.py --config-file $config_file --ckpt $weight_file OUTPUT_DIR $weight_dir TEST.BATCH_SIZE $batch_size

