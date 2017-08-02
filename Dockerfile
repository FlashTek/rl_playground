FROM valohai/keras:2.0.0-tensorflow1.0.1-python3.6-cuda8.0-cudnn5-devel-ubuntu16.04
MAINTAINER Roland Zimmermann <rzrolandzimmermann@gmail.com>

RUN apt-get update && \
    apt-get -y install xvfb zlib1g python-opengl ffmpeg libsdl2-2.0-0 libboost-python1.58.0 libboost-thread1.58.0 libboost-filesystem1.58.0 libboost-system1.58.0 fluidsynth build-essential swig python-dev cmake zlib1g-dev libsdl2-dev libboost-all-dev wget unzip && \
    /usr/local/bin/pip --no-cache-dir install --upgrade 'gym[all]' && \
    dpkg --purge libsdl2-dev libboost-all-dev wget unzip && \
    apt-get -y autoremove && \
    dpkg --purge build-essential swig python-dev cmake zlib1g-dev && \
    apt-get -y autoremove && \
    apt-get clean && \
    rm -r /var/lib/apt/lists/* /root/.cache/pip/

RUN echo '#!/bin/bash' > /tmp/openai-gym.sh
CMD ["/tmp/openai-gym.sh"]
