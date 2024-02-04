docker build -t one-piece-image:devel \
	--build-arg="USERNAME=`(echo $USER)`" \
	--build-arg="USER_UID=`(id -u)`" \
	--build-arg="USER_GID=`(id -g)`" \
	-f docker/Dockerfile .
