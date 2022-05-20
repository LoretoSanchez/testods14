# Optimus Base Image
# image used by containers running internal demonstration applications
# also used as the base image by the ci system for building, testing, deploy etc.

# system dependencies
FROM ubuntu:18.04 AS system

# system dependencies
RUN apt-get update && apt-get install -y build-essential zlib1g-dev libffi-dev libpq-dev software-properties-common ca-certificates

# leverage the ca-certificates ubuntu package to ensure fresh ca certs are in the image
RUN update-ca-certificates --fresh

# utilities and tools. install and setup.
FROM system AS utilities

# useful unix tooling
RUN apt-get update && apt-get install -y git curl rsync lsof jq unzip tree wait-for-it

# bats for shell testing
# (core team development dependency)
RUN git clone https://github.com/sstephenson/bats.git /usr/local/src/bats
RUN /usr/local/src/bats/install.sh /usr/local

# install nodejs for optimus ui builds/development
# kedro ui requires node 10.x^
RUN curl -sL https://deb.nodesource.com/setup_10.x | bash -
RUN apt-get install -y nodejs

# install pandoc to support building latex based documentation
RUN apt-get install -y pandoc

# install anaconda for python development
FROM utilities AS conda-install

# install miniconda to /miniconda
RUN curl -LO https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh

# conda install script and cleanup
RUN bash Miniconda3-latest-Linux-x86_64.sh -p /miniconda -b && rm Miniconda3-latest-Linux-x86_64.sh

# set conda executable
ENV PATH=/miniconda/bin:${PATH}

# update to the latest version of conda if there has been a patch release
RUN conda update -y conda

# update conda to the latest version if it is not already
RUN conda update -n base -c defaults conda

# install anaconda for python development
FROM conda-install AS environment-setup

# set the docker working dir to /optimus
WORKDIR /optimus

# initialize the prefered conda shell. This modifies bashrc to ensure shell context is correct
RUN conda init bash

# create the optimus conda environment
# all optimus python related commands, tests, and long running processes
# should be executed in this conda environment
RUN conda create  -y --name optimus python=3.7 pip

# do not auto activate the base environment
RUN conda config --set auto_activate_base false

# add interactive conda activation command to the bashrc
# ensures that you are in the 'optimus' conda environment when you get a shell in the container
RUN echo "conda activate optimus" >> /root/.bashrc

# install optimus project dependencies
FROM environment-setup AS dependencies

# hotfix
RUN set -m

# install poetry for python dependency management
RUN conda install virtualenv
RUN /bin/bash -c "curl -sSL https://raw.githubusercontent.com/python-poetry/poetry/master/install-poetry.py | python"

# hotfix poetry path
RUN export PATH=/root/.poetry/bin:$PATH

# ensure poetry setup properly for each new bash shell
RUN echo "source $HOME/.local/env" >> /root/.bashrc

# do not allow poetry to create a new virtual environment
RUN /bin/bash -c "/root/.local/bin/poetry config virtualenvs.create false"

# optimus conda env visible to poetry
#RUN /bin/bash -c "/root/.local/bin/poetry env info | grep -q optimus"