FROM sdp-docker-registry.kat.ac.za:5000/docker-base-gpu-build:latest as build
MAINTAINER sperkins@ska.ac.za

# Switch to root for package install
USER root

ENV PACKAGES \
    python-pip \
    curl \
    wget \
    build-essential \
    gfortran \
    libglib2.0-dev \
    libncurses5-dev \
    libreadline-dev \
    flex \
    bison \
    libblas-dev \
    liblapacke-dev \
    libcfitsio-dev \
    # Needs GSL 1.x but bionic has 2.x
    # Manually download and install below
    # libgsl0-dev \
    libfftw3-dev \
    libmotif-dev \
    # Without libcurl Obit pretends it can't find an external xmlrpc
    libcurl4-openssl-dev \
    libxmlrpc-core-c3-dev \
    libxmlrpc-c++8-dev \
    libboost-all-dev \
    subversion \
    # Required by bnmin1
    swig \
    zlib1g-dev

# Update, upgrade and install packages
RUN apt-get update && \
    apt-get install -y $PACKAGES

# Get CUDA samples
RUN CUDA_RUN_FILE=cuda_10.0.130_410.48_linux && \
    wget --progress=dot:mega "http://sdp-services.kat.ac.za/mirror/developer.nvidia.com/compute/cuda/10.0/Prod/local_installers/$CUDA_RUN_FILE" && \
    sh ./$CUDA_RUN_FILE --samples --silent && \
    mv /root/NVIDIA_CUDA-10.0_Samples /usr/local/cuda/samples

ENV KATHOME=/home/kat

# Install gsl 1.16
RUN mkdir -p $KATHOME/src && \
    cd $KATHOME/src && \
    curl ftp://ftp.gnu.org/gnu/gsl/gsl-1.16.tar.gz | tar xzf - && \
    cd gsl-1.16 && \
    ./configure --prefix=/usr && \
    make -j 8 all && \
    make -j 8 install && \
    make DESTDIR=/installs install-strip

# Now downgrade to kat
USER kat

ENV OBIT_REPO https://github.com/bill-cotton/Obit/trunk/ObitSystem
ENV OBIT_BASE_PATH=/home/kat/Obit
ENV OBIT=/home/kat/Obit/ObitSystem/Obit

# Retrieve Obit r592
RUN mkdir -p $OBIT_BASE_PATH && \
    svn co -q -r 592 $OBIT_REPO ${OBIT_BASE_PATH}/ObitSystem

WORKDIR $OBIT_BASE_PATH

# Add OBIT patch
COPY --chown=kat:kat obit.patch /tmp/obit.patch

# Apply OBIT patch
RUN patch -p1 -N -s < /tmp/obit.patch

# Compile Obit
RUN cd ObitSystem/Obit && \
    ./configure --prefix=/usr --without-plplot --without-wvr && \
    make clean && \
    make -j 8

# Compile ObitTalk
# Useful, but not critical tool for interacting with the AIPS filesystem
# This could be removed from the Dockerfile but is useful for debugging
RUN cd ObitSystem/ObitTalk && \
    # --with-obit doesn't pick up the PYTHONPATH and libObit.so correctly
    export PYTHONPATH=$OBIT/python && \
    export LD_LIBRARY_PATH=$OBIT/lib && \
    ./configure --bindir=/bin --with-obit=$OBIT && \
    # Run the main makefile. This gets some of the way but falls over
    # due to lack of latex
    { make || true; }

# Go back to root priviledges to install
# ObitTalk in the /installs filesystem
USER root

# Install ObitTalk
RUN cd ObitSystem/ObitTalk && \
    # Run the main makefile. This gets some of the way but falls over
    # due to lack of latex
    { make DESTDIR=/installs install || true; } && \
    # Just install ObitTalk and ObitTalkServer
    cd bin && \
    make clean && \
    make && \
    make DESTDIR=/installs install

USER kat

# Add python package requirements
COPY --chown=kat:kat katacomb/requirements.txt /tmp/requirements.txt

# Install required python packages
ENV PATH="$PATH_PYTHON2" VIRTUAL_ENV="$VIRTUAL_ENV_PYTHON2"
RUN install-requirements.py -d ~/docker-base/base-requirements.txt -r /tmp/requirements.txt

# Install katacomb
COPY --chown=kat:kat katacomb $KATHOME/src/katacomb

RUN pip install $KATHOME/src/katacomb

#######################################################################

FROM sdp-docker-registry.kat.ac.za:5000/docker-base-gpu-runtime:latest
MAINTAINER sperkins@ska.ac.za

# Switch to root for package install
USER root

ENV PACKAGES \
    libglib2.0-0 \
    libncurses5 \
    libreadline7 \
    libcurl4 \
    libxmlrpc-core-c3 \
    libxmlrpc-c++8v5 \
    libxm4

RUN apt-get update && \
    apt-get install -y --no-install-recommends $PACKAGES && \
    rm -rf /var/lib/apt/lists/*

# Now downgrade to kat
USER kat

# Install system packages
COPY --from=build /installs /
COPY --from=build --chown=kat:kat /home/kat/Obit /home/kat/Obit

# Add task configuration files
COPY --chown=kat:kat katacomb/katacomb/conf /obitconf

# Install Python ve
COPY --from=build --chown=kat:kat /home/kat/ve /home/kat/ve
ENV PATH="$PATH_PYTHON2" VIRTUAL_ENV="$VIRTUAL_ENV_PYTHON2"

# Set up Obit environment
ENV OBIT_BASE_PATH=/home/kat/Obit
ENV OBIT="$OBIT_BASE_PATH"/ObitSystem/Obit \
    OBITINSTALL="$OBIT_BASE_PATH" \
    OBIT_EXEC="$OBIT" \
    OBITSD="$OBIT_BASE_PATH"/ObitSystem/ObitSD
ENV PATH="$OBIT_BASE_PATH"/ObitSystem/Obit/bin:"$PATH"
ENV LD_LIBRARY_PATH="$OBIT_BASE_PATH"/ObitSystem/Obit/lib
ENV PYTHONPATH=/usr/local/share/obittalk/python
ENV PYTHONPATH="$PYTHONPATH":$OBIT_BASE_PATH/ObitSystem/Obit/python
ENV PYTHONPATH="$PYTHONPATH":$OBIT_BASE_PATH/ObitSystem/ObitSD/python

# Set the work directory to /obitconf
WORKDIR /obitconf

# Configure Obit/AIPS disks
RUN cfg_aips_disks.py

# Execute test cases
RUN nosetests katacomb
