#!/bin/bash

docker run -itd --gpus all \
	--name one-piece-recongnition-prod \
	-p 8080:80 \
	-v /etc/localtime:/etc/localtime:ro \
	-v /etc/timezone:/etc/timezone:ro \
	one-piece-image:prod
