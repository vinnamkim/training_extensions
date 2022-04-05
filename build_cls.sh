TAG=v0.1.1

docker build -t mmcls:$TAG . -f Dockerfile.mmcls
docker build -t otecls:$TAG . -f Dockerfile.otecls --build-arg TAG=$TAG
