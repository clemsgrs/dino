FROM doduo1.umcn.nl/uokbaseimage/base:tf2.10-pt1.12

USER root
WORKDIR /home/user/

# install pyvips
RUN apt-key adv --fetch-keys https://developer.download.nvidia.com/compute/cuda/repos/ubuntu1804/x86_64/3bf863cc.pub

# download & unpack vips
RUN wget -q https://github.com/libvips/libvips/releases/download/v8.13.0/vips-8.13.0.tar.gz  && \
    sudo -S apt-get update && \
    sudo -S apt-get install -y libjpeg-turbo8-dev && \
    sudo -S apt-get install -y libgtk2.0-dev && \
    sudo -S apt-get install -y libgsf-1-dev && \
    sudo -S apt-get install -y libtiff5-dev && \
    sudo -S apt-get install -y libopenslide-dev && \
    tar xf vips-8.13.0.tar.gz > /dev/null 2>&1

# sepcify workdir
WORKDIR /home/user/vips-8.13.0/

# make and install vips
RUN ./configure && \
    make && \
    sudo -S make install && \
    cd .. && \
    sudo -S ldconfig && \
    sudo -S pip3.9 install pyvips

WORKDIR /home/user/
RUN rm vips-8.13.0.tar.gz

# install ASAP
COPY ASAP*.deb .
RUN apt-get update && \
    apt-get install --assume-yes /home/user/ASAP-2.1-Ubuntu2004.deb && \
    ldconfig && \
    SITE_PACKAGES=`python3 -c "import sysconfig; print(sysconfig.get_paths()['purelib'])"` && \
    printf "/opt/ASAP/bin/\n" > "${SITE_PACKAGES}/asap.pth" && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/*

RUN rm ASAP*.deb

USER user

# clone dino repo and install requirements
RUN git clone https://github.com/computationalpathologygroup/dino.git
RUN pip3 install -r dino/requirements.txt`

#### Configure entrypoint
COPY run.sh .
ENTRYPOINT ["/bin/bash", "/home/user/run.sh"]