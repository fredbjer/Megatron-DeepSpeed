#! /bin/bash

# Runs the "145B" parameter model
# 145B- for 80 layers -> 78 layers 

export NCCL_IB_GID_INDEX=0
export NCCL_SOCKET_IFNAME=bond0 
export NCCL_IB_DISABLE=0
export NCCL_DEBUG=INFO

GPUS_PER_NODE=8
# Change for multinode config
MASTER_ADDR=10.119.61.12
MASTER_PORT=6000
NNODES=6
NODE_RANK=0
WORLD_SIZE=$(($GPUS_PER_NODE*$NNODES))

DATA_PATH=/data/gpt_data/my-gpt2_text_document
VOCAB_PATH=/data/gpt_data/vocabs
CHECKPOINT_PATH=/data/checkpoints

DISTRIBUTED_ARGS="--nproc_per_node $GPUS_PER_NODE --nnodes $NNODES --node_rank $NODE_RANK --master_addr $MASTER_ADDR --master_port $MASTER_PORT"

torchrun $DISTRIBUTED_ARGS \
       pretrain_gpt.py \
       --tensor-model-parallel-size 8 \
       --pipeline-model-parallel-size 6 \
       --num-layers 78 \
       --hidden-size 12288 \
       --num-attention-heads 64 \
       --micro-batch-size 4 \
       --global-batch-size 768 \
       --seq-length 1024 \
       --max-position-embeddings 1024 \
       --train-iters 500000 \
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
       --log-interval 1 \
       --save-interval 10 \
       --eval-interval 10 \
       --eval-iters 10 \
       --bf16
