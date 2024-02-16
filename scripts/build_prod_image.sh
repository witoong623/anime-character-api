#!/bin/bash

docker build -t one-piece-image:prod \
	--target prod \
	-f docker/Dockerfile .
