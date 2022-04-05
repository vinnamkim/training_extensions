docker run --gpus all -d -p 8888:8888 \
    --name=ote-jupyter \
    --ipc=host \
    -v /local/dataset:/dataset \
    -v /home/vinnamki/outputs:/outputs \
    -v torch-cache:/root/.cache/torch \
    -v torch-model:/root/.torch/models \
    otedet-jupyter:v0.1.1
