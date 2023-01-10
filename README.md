# Stable Diffusion Training Pipeline

## Instructions

### Set up environment

```bash
make environment
```

A virtual environment will be created and the required python dependencies will be installed.

### Training a Model

Edit `config` and define the following:
- `MODEL_NAME` - Model you would like to train (from huggingface or a path)
- `INSTANCE_DIR` - Directory where your training images are located
- `CLASS_DIR` - Directory you would like to store generated class images
- `OUTPUT_DIR` - Directory you would like to save models and checkpoints
- `INSTANCE_PROMPT` - Prompt you would like to train with
- `CLASS_PROMPT` - Prompt to generate class images

Once you are happy with the config, run the following to train:

```bash
make train
```

If you'd like to continue training from a checkpoint, uncomment `--resume_from_checkpoint` and set the value to the checkpoint you're interested in and re-run

```bash
make train
```

### Testing the Model

To test a trained model, start Jupyter server, then run the `test_trained_model` notebook

```bash
make jupyter
```

You will need to copy the URL and Token and paste it into your browser.

