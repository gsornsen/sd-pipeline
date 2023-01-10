PROJ_DIR="`pwd`"
SYSTEM_PYTHON=python3
VENV_PYTHON="${PROJ_DIR}/venv/bin/python3"
SOURCE_DIR="${PROJ_DIR}"

virtualenv:
	@echo "Installing virtualenv if necessary"
	@${SYSTEM_PYTHON} -m pip install virtualenv
	@${SYSTEM_PYTHON} -m pip install --upgrade pip
    
venv:
	@echo "Creating virtual environment"
	@${SYSTEM_PYTHON} -m virtualenv venv

deps:
	@echo "Installing python dependencies"
	@${VENV_PYTHON} -m pip install -r requirements.txt
	@${VENV_PYTHON} -m pip install git+https://github.com/huggingface/diffusers
    
environment: virtualenv venv deps
	@echo "Setting up the python environment"
	@mkdir -p images
	@mkdir -p models
    
train:
	@echo "Starting Training"
	@./train.sh

jupyter:
	@echo "Starting Jupyter Server. You will need to open another console to do more"
	@${VENV_PYTHON} -m jupyter lab --no-browser --ip="*"


clean:
	@rm -rf venv

.PHONY: virtualenv venv deps environment train jupyter clean

