FROM ubuntu:latest
MAINTAINER Ashik Vetrivelu <vcashik@gmail.com>

RUN apt-get update
RUN apt-get install --no-install-recommends --no-install-suggests -y curl

RUN apt-get install --no-install-recommends -y python3-dev \
   git \
   wget \
   liblapack-dev \
   libopenblas-dev \
   python3-numpy \
   build-essential \
   sudo \
   cmake \
   clang-3.6 \
   clang-format-3.6 \
   python3-pip \
   python3-nose \
   python3-wheel \
   python3-setuptools \
   python3-scipy

RUN pip3 install --upgrade pip

RUN pip3 install --upgrade numpy

RUN pip3 install --upgrade Cython

RUN cd /root && wget http://www.cmake.org/files/v3.8/cmake-3.8.1.tar.gz && \
  tar -xzf cmake-3.8.1.tar.gz && cd cmake-3.8.1 && \
  ./configure && \
  make -j "$(nproc)" && \
  make install

RUN cd /root && git clone https://github.com/Theano/libgpuarray.git && cd libgpuarray && \
   mkdir Build && cd Build && \
   cmake .. -DCMAKE_BUILD_TYPE=Release -DCMAKE_INSTALL_PREFIX=/usr && \
   make -j "$(nproc)" && \
   make install

RUN cd /root/libgpuarray && \
   python3 setup.py build_ext -L /usr/lib -I /usr/include && \
   python3 setup.py install

RUN pip3 install --upgrade pip
RUN pip3 install --upgrade six
RUN pip3 install --upgrade --no-deps git+git://github.com/Theano/Theano.git
RUN pip3 install --upgrade keras
RUN pip3 install --upgrade pillow

WORKDIR /scene_detection

RUN mkdir -p /root/.keras

RUN mkdir -p models

ADD docker/keras.json /root/.keras/

ADD core core/
ADD dataset dataset/

ADD test/ test/

ADD main.py .

ENV LD_LIBRARY_PATH /usr/lib:$LD_LIBRARY_PATH
ENV BLAS /usr/lib/libopenblas.a
ENV ATLAS /usr/lib/libopenblas.a

RUN echo "[blas]\nldflags = -lopenblas\n" >> /root/.theanorc

ENTRYPOINT ["python3","main.py"]
