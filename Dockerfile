# Base image
# Change base image below so that it matches CUDA version on host
FROM nvidia/cuda:11.8.0-base-ubuntu20.04
FROM python:3.8

WORKDIR /home
RUN mkdir researcher
WORKDIR /home/researcher

ADD . /home/researcher/Remix
WORKDIR /home/researcher/Remix

ENV PYTHONPATH="/home/researcher/Remix/TFDM"

RUN pip install -r requirements.txt

