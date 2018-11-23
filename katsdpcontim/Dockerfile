FROM sdp-docker-registry.kat.ac.za:5000/docker-base-build:latest as build
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
    libcfitsio3-dev \
    # Needs GSL 1.x but xenial has 2.x
    # Manually download and install below
    # libgsl0-dev \
    libfftw3-dev \
    libmotif-dev \
    # Without libcurl Obit pretends it can't find an external xmlrpc
    libcurl4-openssl-dev \
    libxmlrpc-core-c3-dev \
    libxmlrpc-c++8-dev \
    libboost-all-dev \
    # Required by bnmin1
    swig \
    zlib1g-dev

# Update, upgrade and install packages
RUN apt-get update && \
    apt-get install -y $PACKAGES

ENV KATHOME=/home/kat
ENV OBIT_BASE_PATH=/home/kat/Obit
ENV OBIT=/home/kat/Obit/ObitSystem/Obit

# Install gsl 1.16
RUN mkdir -p $KATHOME/src && \
    cd $KATHOME/src && \
    curl ftp://ftp.gnu.org/gnu/gsl/gsl-1.16.tar.gz | tar xzvf - && \
    cd gsl-1.16 && \
    ./configure --prefix=/usr && \
    make -j 8 all && \
    make -j 8 install && \
    make DESTDIR=/installs install-strip

# Add python package requirements
COPY --chown=kat:kat requirements.txt /tmp/install-requirements.txt

# Add OBIT patch
COPY --chown=kat:kat obit.patch /tmp/obit.patch

# Now downgrade to kat
USER kat

# Obit r588
ENV OBIT_TARBALL https://api.github.com/repos/bill-cotton/Obit/tarball/62d49e5e7c04cd230cb545389b19cc05b431d7b8

# Download Obit tarball and untar
RUN mkdir -p $OBIT_BASE_PATH && \
    cd $OBIT_BASE_PATH && \
    curl -L $OBIT_TARBALL | tar xz --strip=1

WORKDIR $OBIT_BASE_PATH

# Apply OBIT patch
RUN patch -p1 -N -s < /tmp/obit.patch

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
    # --with-obit doesn't pick up the PYTHONPATH and libObit.so correctly
    export PYTHONPATH=$OBIT/python && \
    export LD_LIBRARY_PATH=$OBIT/lib && \
    ./configure --bindir=/bin --with-obit=$OBIT && \
    # Run the main makefile. This gets some of the way but falls over
    # due to lack of latex
    { make || true; }

# Go back to root priviledges to install
# ObitView and ObitTalk in the /installs filesystem
USER root

# Install ObitView. Its Makefile.in doesn't support DESTDIR, so the install is
# done manually.
RUN cd ObitSystem/ObitView && \
    mkdir -p /installs/usr/bin && \
    install -s ObitView ObitMess /installs/usr/bin

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

COPY --chown=kat:kat katacomb $KATHOME/src/katacomb

USER kat

# Install required python packages
ENV PATH="$PATH_PYTHON2" VIRTUAL_ENV="$VIRTUAL_ENV_PYTHON2"
RUN install-requirements.py -d ~/docker-base/base-requirements.txt -r /tmp/install-requirements.txt

# Install katacomb
RUN pip install $KATHOME/src/katacomb

#######################################################################

FROM sdp-docker-registry.kat.ac.za:5000/docker-base-runtime:latest
MAINTAINER sperkins@ska.ac.za

# Switch to root for package install
USER root

ENV PACKAGES \
    libglib2.0-0 \
    libncurses5 \
    libreadline6 \
    libcurl3 \
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
ENV OBIT_BASE_PATH=/home/kat/Obit  \
    OBIT="$OBIT_BASE_PATH"/ObitSystem/Obit \
    OBITINSTALL="$OBIT_BASE_PATH" \
    OBIT_EXEC="$OBIT" \
    OBITSD=$OBIT_BASE_PATH/ObitSystem/ObitSD
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
