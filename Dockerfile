FROM nvidia/cuda:11.8.0-cudnn8-runtime-ubuntu22.04

# install python
RUN apt-get update && apt-get install -y \
	python3-pip \
	&& rm -rf /var/lib/apt/lists/*

# create virtual environment
RUN pip3 install virtualenv
ENV VIRTUAL_ENV="/opt/venv"
RUN virtualenv -p $(which python3) $VIRTUAL_ENV
ENV PATH="$VIRTUAL_ENV/bin:$PATH"

# install the required dependencies
COPY requirements.txt /root/requirements.txt
RUN pip install -r /root/requirements.txt

# add commands to bashrc (activating virtualenv and cd into working dir)
COPY scripts/bashrc.sh /root/bashrc.sh
RUN cat /root/bashrc.sh >> /root/.bashrc
