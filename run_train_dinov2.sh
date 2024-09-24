#!/bin/bash

# python dinov2/run/train/train.py \
#     --nodes 4 \
#     --config-file dinov2/configs/train/vitl16_short.yaml \
#     --output-dir <PATH/TO/OUTPUT/DIR> \
#     train.dataset_path=ImageNet:split=TRAIN:root=<PATH/TO/DATASET>:extra=<PATH/TO/DATASET>



export CUDA_VISIBLE_DEVICES=0,1 && export PYTHONPATH=. && python -m torch.distributed.launch --nproc_per_node=2 --master_port=12366 dinov2/train/train.py \
    --config-file=dinov2/configs/train/vitb16.yaml \
    --output-dir output/vitb16 \
    train.dataset_path=MyDataSet:root=/path/to/my/dataset:split=train \
    train.batch_size_per_gpu=256 \
    MODEL.WEIGHTS=output/vitb16_old/model_0116249.rank_0.pth \
    no_resume=False