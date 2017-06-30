#!/bin/bash
OBIT_BASE_PATH=/src/Obit

cd $OBIT_BASE_PATH
patch -p0 -N --dry-run --silent < /obit.patch 2>/dev/null
#If the patch has not been applied then the $? which is the exit status
#for last command would have a success status code = 0
if [ $? -eq 0 ];
then
    #apply the patch
    patch -p0 -N < /obit.patch
fi

cd $OBIT_BASE_PATH/trunk/ObitSystem/Obit
./configure --prefix=/usr --without-plplot --without-wvr
make -j 8

export LD_LIBRARY_PATH=$OBIT_BASE_PATH/trunk/ObitSystem/Obit/lib:$LD_LIBRARY_PATH
export PYTHONPATH=$OBIT_BASE_PATH/trunk/ObitSystem/Obit/python:$PYTHONPATH

