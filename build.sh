TAG=v0.1.1

docker build -t mmdet:$TAG . -f Dockerfile.mmdet
docker build -t otedet:$TAG . -f Dockerfile.otedet --build-arg TAG=$TAG
docker build -t otedet-jupyter:$TAG . -f Dockerfile.otedet-jupyter --build-arg TAG=$TAG
