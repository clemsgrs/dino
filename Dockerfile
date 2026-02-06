ARG UBUNTU_VERSION=24.04
ARG CUDA_MAJOR_VERSION=12.8.0
ARG PYTORCH_VERSION=2.7.1

FROM nvidia/cuda:${CUDA_MAJOR_VERSION}-runtime-ubuntu${UBUNTU_VERSION}

# propagate build args
ARG CUDA_MAJOR_VERSION
ARG PYTORCH_VERSION

ARG USER_UID=1001
ARG USER_GID=1001

# ensures that Python output to stdout/stderr is not buffered: prevents missing information when terminating
ENV PYTHONUNBUFFERED=1
ENV DEBIAN_FRONTEND=noninteractive TZ=Europe/Amsterdam

USER root

RUN groupadd --gid ${USER_GID} user \
    && useradd -m --no-log-init --uid ${USER_UID} --gid ${USER_GID} user

# create input/output directory
RUN mkdir /input /output && \
    chown user:user /input /output

# set /home/user as working directory
WORKDIR /home/user
ENV PATH="/home/user/.local/bin:${PATH}"

RUN apt-get update && apt-get install -y --no-install-recommends \
    curl \
    vim screen \
    zip unzip \
    git \
    openssh-server \
    python3-pip python3-dev python-is-python3 \
    python3-venv \
    && mkdir /var/run/sshd \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

RUN chown -R user:user /opt
RUN chmod -R 775 /opt

# switch to user
USER user

# create and use a virtualenv
ENV VIRTUAL_ENV=/opt/venv
RUN python -m venv $VIRTUAL_ENV
ENV PATH="$VIRTUAL_ENV/bin:$PATH"

WORKDIR /opt/app/

# you can add any Python dependencies to requirements.in
RUN python -m pip install --upgrade "pip<25.3" "setuptools" "wheel" \
    && python -m pip install --upgrade "pip-tools" \
    && rm -rf /home/user/.cache/pip

# install requirements
COPY --chown=user:user requirements.in /opt/app/requirements.in
RUN CUDA_IDENTIFIER_PYTORCH=`echo "cu${CUDA_MAJOR_VERSION}" | sed "s|\.||g" | cut -c1-5` && \
    sed -i -e "s|%PYTORCH_VERSION%|${PYTORCH_VERSION}|g" requirements.in && \
    python -m piptools compile requirements.in --verbose \
      --index-url https://pypi.org/simple \
      --extra-index-url https://download.pytorch.org/whl/${CUDA_IDENTIFIER_PYTORCH} && \
    python -m piptools sync && \
    rm -rf ~/.cache/pip*

# expose port for ssh and jupyter
EXPOSE 22 8888