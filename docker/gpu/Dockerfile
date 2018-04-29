FROM nvidia/cuda:9.1-cudnn7-devel-ubuntu16.04
MAINTAINER Ashik Vetrivelu <vcashik@gmail.com>

RUN apt-get update
RUN apt-get install --no-install-recommends --no-install-suggests -y curl

ENV NVIDIA_RUN http://us.download.nvidia.com/tesla/390.46/NVIDIA-Linux-x86_64-390.46.run
ENV CUDA_RUN https://developer.nvidia.com/compute/cuda/9.1/Prod/local_installers/cuda_9.1.85_387.26_linux

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
   kmod \
   python3-scipy

RUN pip3 install --upgrade pip

RUN pip3 install --upgrade numpy

RUN pip3 install --upgrade Cython

RUN cd /root && wget http://www.cmake.org/files/v3.8/cmake-3.8.1.tar.gz && \
  tar -xzf cmake-3.8.1.tar.gz && cd cmake-3.8.1 && \
  ./configure && \
  make -j "$(nproc)" && \
  make install

ENV CUDA_ROOT /usr/local/cuda/bin

# RUN cd /root && wget $CUDA_RUN && \
#    chmod +x cuda_9.1.85_387.26_linux && \
#    mkdir -p /root/nvidia_installers && \
#    ./cuda_9.1.85_387.26_linux -extract=`pwd`/nvidia_installers && \
#    cd /root/nvidia_installers && \
#    ./NVIDIA_zLinux-x86_64-387.26.run -s -N --no-kernel-module
#
# RUN cd /root/nvidia_installers && \
#    ./cuda-linux.9.1.85-23083092.run -noprompt
#
# ENV LIBRARY_PATH $LD_LIBRARY_PATH:/usr/local/cuda-9.1/lib64
# ENV PATH $PATH:/usr/local/cuda-9.1/bin
#
# RUN cd /root/nvidia_installers && \
#    ./cuda-samples.9.1.85-23083092-linux.run -noprompt -cudaprefix=/usr/local/cuda-9.1 && \
#    cd /usr/local/cuda/samples/1_Utilities/deviceQuery && \
#    make

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
RUN pip3 install deepdish

WORKDIR /scene_detection

RUN mkdir -p /root/.keras

RUN mkdir -p models

ADD docker/gpu/keras.json /root/.keras/

ADD core core/
ADD dataset/imagenet.h5 dataset/imagenet.h5

ADD test/ test/

ADD main.py .

ENV LD_LIBRARY_PATH /usr/lib:$LD_LIBRARY_PATH
ENV BLAS /usr/lib/libopenblas.a
ENV ATLAS /usr/lib/libopenblas.a

RUN rm -rf /tmp/*

RUN echo '[global]\ndevice=cuda0\nfloatX=float32\nwintX=int32\noptimizer_including=cudnn\n\
[cuda]\nroot=/usr/local/cuda-9.1/include\n\
[blas]\nldflags = -lopenblas\n\
[dnn]\nenable=True\nlibrary_path=/usr/local/cuda-9.1/lib64\ninclude_path=/usr/local/cuda-9.1/include' >> /root/.theanorc

ENTRYPOINT ["python3","main.py"]
