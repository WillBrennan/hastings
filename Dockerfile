FROM    gcc:latest

ENV     CC=gcc
ENV     CXX=g++
ENV     DEBIAN_FRONTEND="noninteractive"
ENV     LD_LIBRARY_PATH=/usr/local/lib

RUN     apt update && apt install -yq build-essential git cmake make clang-format libgoogle-glog-dev libgflags-dev

WORKDIR /root/
RUN     git clone --depth 1 --branch 4.9.0 https://github.com/opencv/opencv.git && \
        git clone --depth 1 --branch 4.9.0 https://github.com/opencv/opencv_contrib
RUN     mkdir -p /root/opencv/build &
WORKDIR /root/opencv/build
RUN     cmake -DWITH_JPEG=ON -DOPENCV_EXTRA_MODULES_PATH=/root/opencv_contrib/modules -DBUILD_opencv_dnn=OFF ../ && \
        make -j`nproc` && make install

RUN     apt install -yq libboost-filesystem-dev libboost-system-dev libboost-thread-dev libboost-regex-dev

ADD     . /root/hastings
RUN     mkdir -p /root/hastings/build
WORKDIR /root/hastings/build
RUN     rm -rf * && cmake ../ && make -j`nproc` && make test