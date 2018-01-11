FROM sdp-docker-registry.kat.ac.za:5000/docker-base:latest
MAINTAINER sperkins@ska.ac.za

# Switch to root for package install
USER root

ENV DEBIAN_FRONTEND noninteractive

ENV PACKAGES \
    software-properties-common \
    python-software-properties \
    python-pip \
    curl \
    vim \
    wget \
    git \
    cvs \
    subversion \
    autotools-dev \
    automake \
    build-essential \
    cmake \
    gfortran \
    g++ \
    libglib2.0-dev \
    libncurses5-dev \
    libreadline-dev \
    flex \
    bison \
    libblas-dev \
    liblapacke-dev \
    libcfitsio3-dev \
    # Needs GSL 1.x but xenial has 2.x
    # Manually download and install below
    # libgsl0-dev \
    wcslib-dev \
    libhdf5-serial-dev \
    libfftw3-dev \
    python-numpy \
    libmotif-dev \
    # Without libcurl Obit pretends it can't find an external xmlrpc
    libcurl4-openssl-dev \
    libxmlrpc-core-c3-dev \
    libxmlrpc-c++8-dev \
    libboost-all-dev \
    # Required by bnmin1
    swig \
    zlib1g-dev \
    libpython3.5-dev \
    libpython2.7-dev \
    python-tk

# Update, upgrade and install packages
RUN apt-get update && \
    apt-get upgrade -y && \
    apt-get install -y $PACKAGES && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/* /tmp/* /var/tmp/*

# Install gsl 1.16
RUN mkdir -p /src && \
    cd /src && \
    curl ftp://ftp.gnu.org/gnu/gsl/gsl-1.16.tar.gz | tar xzvf - && \
    cd gsl-1.16 && \
    ./configure --prefix=/usr && \
    make -j 8 all && \
    make -j 8 install

ENV OBIT_BASE_PATH /home/kat/Obit
ENV OBIT $OBIT_BASE_PATH/ObitSystem/Obit

# Add task configuration files
ADD katacomb/katacomb/conf /obitconf

# Add the Obit source code
ADD Obit/trunk $OBIT_BASE_PATH

# Add OBIT patch
ADD obit.patch $OBIT_BASE_PATH/obit.patch

# Add OBIT setup script
ADD setup_obit.sh /bin/setup_obit.sh

ADD obit_requirements.txt /tmp/obit_requirements.txt
ADD ve_requirements.txt /tmp/ve_requirements.txt

ADD katacomb /home/kat/src/katacomb

# Ensure everything under /home/kat belongs to kat
RUN chown -R kat:kat /home/kat

# Install obit requirements as root so that packages
# like ObitTalk and ObitView have access to them
RUN pip install -r /tmp/obit_requirements.txt

# Now downgrade to kat
USER kat

# Add obit setup to bashrc
RUN touch /home/kat/.bashrc && \
    cat /bin/setup_obit.sh >> /home/kat/.bashrc

RUN . ~/ve/bin/activate && \
    pip install -U pip setuptools wheel

RUN . ~/ve/bin/activate && \
    pip install -r /tmp/obit_requirements.txt -r /tmp/ve_requirements.txt

RUN . ~/ve/bin/activate && \
    pip install -e /home/kat/src/katacomb

RUN mkdir -p $OBIT_BASE_PATH
WORKDIR $OBIT_BASE_PATH

# Apply OBIT patch
RUN patch -p1 -N -s < obit.patch

# Compile Obit
RUN cd ObitSystem/Obit && \
    ./configure --prefix=/usr --without-plplot --without-wvr && \
    make clean && \
    make -j 8

# Compile ObitView
# Useful, but not critical image viewing utility
# that can interact with Obit MFImage while it is running, or with ObitTalk.
# This could be removed from Dockerfile but is useful for debugging
RUN cd ObitSystem/ObitView && \
    ./configure --prefix=/usr --with-obit=$OBIT --without-plplot --without-wvr && \
    make clean && \
    make

# Compile ObitTalk
# Useful, but not critical tool for interacting with the AIPS filesystem
# and ObitView
# This could be removed from the Dockerfile but is useful for debugging
RUN cd ObitSystem/ObitTalk && \
    # --with-obit doesn'tt pick up the PYTHONPATH and libObit.so correctly
    export PYTHONPATH=$OBIT/python && \
    export LD_LIBRARY_PATH=$OBIT/lib && \
    ./configure --bindir=/bin --with-obit=$OBIT && \
    # Run the main makefile. This gets some of the way but falls over
    # due to lack of latex
    make ||true

# Go back to root priviledges to install
# ObitView and ObitTalk in the /usr filesystem
USER root

# Install ObitView
RUN cd ObitSystem/ObitView && \
    make install

# Install ObitTalk
RUN cd ObitSystem/ObitTalk && \
    # Run the main makefile. This gets some of the way but falls over
    # due to lack of latex
    make install || true && \
    # Just install ObitTalk and ObitTalkServer
    cd bin && \
    make clean && \
    make && \
    make install

USER kat

# Set the work directory to /obitconf
WORKDIR /obitconf

# Configure Obit/AIPS disks
RUN /bin/bash -c ". /bin/setup_obit.sh && cfg_aips_disks.py"