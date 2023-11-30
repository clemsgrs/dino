FROM doduo1.umcn.nl/uokbaseimage/base:tf2.10-pt1.12

USER user
WORKDIR /home/user/

# clone dino repo and install requirements
RUN git clone https://github.com/computationalpathologygroup/dino.git
RUN pip3 install -r dino/requirements.txt

#### Configure entrypoint
COPY run.sh .
ENTRYPOINT ["/bin/bash", "/home/user/run.sh"]