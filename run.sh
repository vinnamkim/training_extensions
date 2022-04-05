nvidia-docker run -d -p 8888:8888 \
    --name=ote-jupyter \
    --ipc=host \
    -v /local_ssd3/dataset:/dataset \
    -v /local_ssd3/vinnamki/outputs:/outputs \
    -v torch-cache:/root/.cache/torch \
    -v torch-model:/root/.torch/models \
    otedet-jupyter:v0.1.1
