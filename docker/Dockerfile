FROM nvcr.io/nvidia/tensorrt:24.01-py3 as devel

ENV TZ=Asia/Bangkok \
    DEBIAN_FRONTEND=noninteractive

RUN apt-get update && apt-get install -y build-essential tmux sudo vim

ARG USERNAME=witoon
ARG USER_UID=1001
ARG USER_GID=1001

RUN groupadd -g $USER_GID $USERNAME

RUN useradd -rm -d /home/$USERNAME -s /bin/bash -g $USER_GID -G sudo -u $USER_UID $USERNAME \
    && echo $USERNAME ALL=\(root\) NOPASSWD:ALL > /etc/sudoers.d/$USERNAME \
    && chmod 0440 /etc/sudoers.d/$USERNAME

USER $USERNAME

WORKDIR /home/$USERNAME


FROM nvcr.io/nvidia/tensorrt:24.01-py3 as prod

ENV TZ=Asia/Bangkok \
    DEBIAN_FRONTEND=noninteractive

WORKDIR /work

COPY requirements.txt /work/requirements.txt

RUN pip3 install --no-cache-dir --upgrade -r /work/requirements.txt

COPY one-piece-classifier-b16-fp16.engine /work/one-piece-classifier-b16-fp16.engine

COPY common.py /work/common.py

COPY main.py /work/main.py

CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "80"]
