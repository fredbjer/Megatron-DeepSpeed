#! /bin/bash

# Runs the "345M" parameter model

GPUS_PER_NODE=8
# Change for multinode config
MASTER_ADDR=10.119.61.15
MASTER_PORT=6000
NNODES=2
NODE_RANK=1
WORLD_SIZE=$(($GPUS_PER_NODE*$NNODES))

DATA_PATH=/data/gpt_data/my-gpt2_text_document
VOCAB_PATH=/data/gpt_data/vocabs
CHECKPOINT_PATH=/data/checkpoints

DISTRIBUTED_ARGS="--nproc_per_node $GPUS_PER_NODE --nnodes $NNODES --node_rank $NODE_RANK --master_addr $MASTER_ADDR --master_port $MASTER_PORT"

python3 -m torch.distributed.launch $DISTRIBUTED_ARGS \
       pretrain_gpt.py \
       --num-layers 24 \
       --hidden-size 1024 \
       --num-attention-heads 16 \
       --micro-batch-size 1 \
       --global-batch-size 64 \
       --seq-length 1024 \
       --max-position-embeddings 1024 \
       --train-iters 500 \
       --lr-decay-iters 320000 \
       --save $CHECKPOINT_PATH \
       --load $CHECKPOINT_PATH \
       --data-path $DATA_PATH \
       --vocab-file $VOCAB_PATH/gpt2-vocab.json \
       --merge-file $VOCAB_PATH/gpt2-merges.txt \
       --data-impl mmap \
       --split 949,50,1 \
       --distributed-backend nccl \
       --lr 0.00015 \
       --lr-decay-style cosine \
       --min-lr 1.0e-5 \
       --weight-decay 1e-2 \
       --clip-grad 1.0 \
       --lr-warmup-fraction .01 \
       --checkpoint-activations \
       --log-interval 10 \
       --save-interval 10 \
       --eval-interval 10 \
       --eval-iters 10 \
       --fp16