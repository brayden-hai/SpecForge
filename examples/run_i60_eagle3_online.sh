SCRIPT_DIR=$( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )
ROOT_DIR=$(dirname $SCRIPT_DIR)
export TORCHINDUCTOR_CACHE_DIR=$ROOT_DIR/cache/compiled_kernels

# train eagle3 for Llama 3.1 405B
NUM_GPUS=8

torchrun \
    --standalone \
    --nproc_per_node $NUM_GPUS \
    $ROOT_DIR/scripts/train_eagle3_online.py \
    --target-model-path /hai-coreweave-training/release/i60 \
    --draft-model-config $ROOT_DIR/configs/llama3-405B-eagle3.json \
    --train-data-path /workspace/brayden/dataset.jsonl \
    --output-dir /workspace/brayden/i60-eagle3 \
    --tp-size 8 \
    --num-epochs 2 \
    --batch-size 1 \
    --learning-rate 1e-4 \
    --max-length 2048 \
    --chat-template i60 \
    --cache-dir $ROOT_DIR/cache \
    --attention-backend flex_attention
