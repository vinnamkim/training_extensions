docker run --gpus all --ipc host -it --rm \
    --name otedet \
    -v /local/dataset:/dataset \
    -v /home/vinnamki/outputs:/outputs \
    -v torch-cache:/root/.cache/torch \
    -v torch-model:/root/.torch/models \
    otecls:v0.1.1 
