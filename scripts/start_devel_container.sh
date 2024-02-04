docker run -itd --gpus all \
	--name one-piece-recongnition \
	-p 8000:8000 \
	-v /etc/localtime:/etc/localtime:ro \
	-v /etc/timezone:/etc/timezone:ro \
	-v $(pwd):/home/witoon/programming-practice/one-piece-api \
	one-piece-image:devel /bin/bash
