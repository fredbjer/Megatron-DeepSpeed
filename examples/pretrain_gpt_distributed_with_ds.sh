#! /bin/bash

# Runs the "345M" parameter model

GPUS_PER_NODE=8
# Change for multinode config
MASTER_ADDR=172.22.84.14
MASTER_PORT=9009
NNODES=12
NODE_RANK=0
WORLD_SIZE=$(($GPUS_PER_NODE*$NNODES))
DATA_PATH=/home/user/data/gpt_data/my-gpt2_text_document
CHECKPOINT_PATH=/home/user/data/checkpoints
VOCAB_PATH=/home/user/data/gpt_data/vocabs

DISTRIBUTED_ARGS="--nproc_per_node $GPUS_PER_NODE --nnodes $NNODES --node_rank $1 --master_addr $MASTER_ADDR --master_port $MASTER_PORT"

DS_CONFIG=ds_config.json

TP=4
PP=12

NHIDDEN=14336
NLAYERS=70
NHEADS=112
SEQ_LEN=2048

GLOBAL_BATCH=384
MICRO_BATCH=4

ZERO_STAGE=1

OUTPUT_DIR=ds_z${ZERO_STAGE}_nl${NLAYERS}_hs${HIDDEN}_gb${GLOBAL_BATCH}_mb${MICRO_BATCH}
#OUTPUT_DIR=baseline_nl${NLAYERS}_hs${HIDDEN}_gb${GLOBAL_BATCH}_mb${MICRO_BATCH}
mkdir -p $OUTPUT_DIR

cat <<EOT > $DS_CONFIG
{
  "train_batch_size" : $GLOBAL_BATCH,
  "train_micro_batch_size_per_gpu": $MICRO_BATCH,
  "steps_per_print": 1,

  "zero_optimization": {
    "stage": $ZERO_STAGE
  },
  "bf16": {
    "enabled": false
  },
  "fp16": {
    "enabled": true,
    "initial_scale_power": 12
  },  
  "wall_clock_breakdown" : true
}
EOT

export NCCL_DEBUG=warn 

ds_args=""
ds_args=" --deepspeed ${ds_args}"
ds_args=" --deepspeed_config=$DS_CONFIG ${ds_args}"
ds_args=" --zero-stage=$ZERO_STAGE ${ds_args}"
ds_args=" --deepspeed-activation-checkpointing ${ds_args}"

torchrun $DISTRIBUTED_ARGS \
       pretrain_gpt.py \
       --tensor-model-parallel-size $TP \
       --pipeline-model-parallel-size $PP \
       --num-layers $NLAYERS \
       --hidden-size  $NHIDDEN\
       --num-attention-heads $NHEADS \
       --micro-batch-size $MICRO_BATCH \
       --global-batch-size $GLOBAL_BATCH \
       --seq-length $SEQ_LEN \
       --max-position-embeddings $SEQ_LEN \
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
       --no-masked-softmax-fusion \
       --no-bias-gelu-fusion \
       --no-bias-dropout-fusion \
       --log-interval 100 \
       --save-interval 10000 \
       --eval-interval 1000 \
       --eval-iters 10 \
       --fp16 \
       --tensorboard-dir $OUTPUT_DIR \
       $ds_args \
       --exit-interval 5000
