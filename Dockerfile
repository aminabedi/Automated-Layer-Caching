FROM pytorch/torchserve 

USER root

WORKDIR /app/caching

COPY requirements.txt /app/caching

RUN apt-get update -y && apt-get install -y curl

RUN pip install -r requirements.txt

COPY . .

RUN mkdir model_store && \
    torch-model-archiver \
    --model-name cifar10-resnet50 \
    --version 1.0 \
    --model-file backbone/cifar10/resnet.py \
    --serialized-file models/cifar10/resnet50.pt \
    --handler  handlers/object_handler.py \
    --export-path model_store \
    --extra-files backbone/cifar10/config.yaml,backbone/modules.py,backbone/cifar10/aux.py,classifier/NConvMDense.py


