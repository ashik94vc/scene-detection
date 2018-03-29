FROM ubuntu:latest
MAINTAINER Ashik Vetrivelu <vcashik@gmail.com>

RUN apt-get update
RUN apt-get install --no-install-recommends --no-install-suggests -y curl

RUN apt-get install --no-install-recommends python \
   git \
   wget \
   liblapack-dev \
   libopenblas-dev \
   python-numpy \
   python-pip \
   python-nose \
   python-scipy

RUN pip install numpy

RUN pip install theano

RUN pip install Cython

RUN cd /root && wget http://www.cmake.org/files/v3.8/cmake-3.8.1.tar.gz && \
  tar -xf cmake-3.8.1.tar.gz && cd cmake-3.8.1 && \
  ./configure && \
  make -j "$(nproc)" && \
  make install

RUN cd /root && git clone https://github.com/Theano/libgpuarray.git && cd libgpuarray && \
   mkdir Build && cd Build
   cmake .. -DCMAKE_BUILD_TYPE=Release -DCMAKE_INSTALL_PREFIX=/usr && \
   make -j"$(nproc)" && \
   make install

RUN cd /root/libgpuarray && \
   python setup.py build_ext -L /usr/lib -I /usr/include && \
   python setup.py install

RUN pip install --upgrade pip
RUN pip install --upgrade six
RUN pip install --upgrade theano
RUN pip install --upgrade keras

WORKDIR /scene_detection

RUN echo ""
