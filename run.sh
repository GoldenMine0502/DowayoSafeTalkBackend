python -m torch.distributed.launch --nproc_per_node=2 main.py \
    --batch_size 16 \
    --num_workers 4 \
    --gpu_devices 0 1\
    --distributed \
    --log_freq 100