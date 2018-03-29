FROM ubuntu:latest
MAINTAINER Ashik Vetrivelu <vcashik@gmail.com>

RUN apt-get update
RUN apt-get install --no-install-recommends --no-install-suggests -y curl

RUN apt-get install --no-install-recommends python

RUN pip install numpy

RUN pip install theano

RUN pip install Cython
