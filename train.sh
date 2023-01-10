#!/usr/bin/env bash
set -e

# Checked a working version into the repo. The command below will grab the latest from the
# main branch. It had low precision detection that crashed out on GPUs with < 12 gigs of VRAM
# and using accelerate to offload optimization steps to the CPU to reduce the burden
# of VRAM usage
# wget "https://raw.githubusercontent.com/huggingface/diffusers/main/examples/dreambooth/train_dreambooth.py"

# Retrieve the GPU information
GPU_INFO="$(nvidia-smi --query-gpu=name,memory.total --format=csv)"
# Parse the csv string and store the values in variables
GPU_NAME=$(echo "$GPU_INFO" | awk -F', ' 'NR==2{print $1}')
GPU_MEM_MB=$(echo "$GPU_INFO" | awk -F', ' 'NR==2{print $2}' | sed 's/[^0-9]*//g')
GPU_MEM_GB=$(echo $((${GPU_MEM_MB} / 1024)))

# Get config values
source config
# Print the config to console
printf "Model Name/Path: ${MODEL_NAME}\nInstance Image Directory: ${INSTANCE_DIR}\nClass Image Directory: ${CLASS_DIR}\nOutput Directory ${OUTPUT_DIR}\n"

# Make parent directories from config if they don't exist
mkdir -p "${INSTANCE_DIR}"
mkdir -p "${CLASS_DIR}"
mkdir -p "${OUTPUT_DIR}"

function remove_jupyter_checkpoints {
    # Pillow raises an IsADirectoryError during training after
    # generating class images if folders are present. Jupyter notebooks
    # drop checkpoint folders all over the filesystem, let's clean
    # these up and save us a bunch of time from having to restart
    # training after class images are generated
    find . -type d -name ".ipynb_checkpoints" -exec rm -rf {} \;
}

function dispatch_based_on_vram_size {
    # Call the 
    if [ ${GPU_MEM_GB} -gt 23 ]; then
    twenty_four_gigs_vram
    elif [ ${GPU_MEM_GB} -gt 12 ]; then
    sixteen_gigs_vram
    elif [ ${GPU_MEM_GB} -le 12 ]; then
    eight_gigs_vram
    else
    echo "Unknown GPU Size: ${GPU_MEM_GB}"
    fi
}

function twenty_four_gigs_vram {
    # 24GB VRAM
    remove_jupyter_checkpoints
    # Training with prior preservation for faces at full precision and also training the text encoder
    accelerate launch train_dreambooth.py \
      --pretrained_model_name_or_path=$MODEL_NAME  \
      --train_text_encoder \
      --instance_data_dir=$INSTANCE_DIR \
      --class_data_dir=$CLASS_DIR \
      --output_dir=$OUTPUT_DIR \
      --with_prior_preservation --prior_loss_weight=1.0 \
      --instance_prompt=$INSTANCE_PROMPT \
      --class_prompt=$CLASS_PROMPT \
      --resolution=768 \
      --train_batch_size=1 \
      --use_8bit_adam
      --gradient_checkpointing \
      --learning_rate=2e-6 \
      --lr_scheduler="constant" \
      --lr_warmup_steps=0 \
      --num_class_images=200 \
      --max_train_steps=3000 \
      --checkpointing_steps=250 \
      # --resume_from_checkpoint="checkpoint-2000"
}

function sixteen_gigs_vram {
    # 16GB VRAM
    remove_jupyter_checkpoints
    # Training with prior preservation for faces at full precsion, but not the text encoder
    accelerate launch train_dreambooth.py \
      --pretrained_model_name_or_path=$MODEL_NAME  \
      --instance_data_dir=$INSTANCE_DIR \
      --class_data_dir=$CLASS_DIR \
      --output_dir=$OUTPUT_DIR \
      --with_prior_preservation --prior_loss_weight=1.0 \
      --instance_prompt=$INSTANCE_PROMPT \
      --class_prompt=$CLASS_PROMPT \
      --resolution=768 \
      --train_batch_size=1 \
      --use_8bit_adam
      --gradient_checkpointing \
      --learning_rate=2e-6 \
      --lr_scheduler="constant" \
      --lr_warmup_steps=0 \
      --num_class_images=200 \
      --max_train_steps=3000 \
      --checkpointing_steps=250 \
      # --resume_from_checkpoint="checkpoint-2000"
}

function eight_gigs_vram {
    # 8GB VRAM
    remove_jupyter_checkpoints
    # Training with prior preservation at half precision, but not the text encoder
    # Includes optimization, including using xformers
    
    # NOTE: This repo does not currently handle installing xformers as it
    # requires compilation and I have only been able to get it to work
    # using conda and a bunch of manual intervention
    accelerate launch --num_cpu_threads_per_process=8 train_dreambooth.py \
      --pretrained_model_name_or_path=$MODEL_NAME \
      --instance_data_dir=$INSTANCE_DIR \
      --class_data_dir=$CLASS_DIR \
      --output_dir=$OUTPUT_DIR \
      --with_prior_preservation --prior_loss_weight=1.0 \
      --prior_generation_precision="fp16" \
      --instance_prompt=$INSTANCE_PROMPT \
      --class_prompt=$CLASS_PROMPT \
      --resolution=512 \
      --train_batch_size=1 \
      --sample_batch_size=1 \
      --gradient_accumulation_steps=1 --gradient_checkpointing \
      --learning_rate=5e-6 \
      --lr_scheduler="constant" \
      --lr_warmup_steps=0 \
      --num_class_images=200 \
      --max_train_steps=3000 \
      --checkpointing_steps=250 \
      --mixed_precision="fp16" \
      --enable_xformers_memory_efficient_attention \
      # --resume_from_checkpoint="checkpoint-2000"

}


dispatch_based_on_vram_size

