#!/bin/bash
export OBIT_BASE_PATH=/src/Obit
export OBIT=$OBIT_BASE_PATH/trunk/ObitSystem/Obit


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

# So we can import libObit.so
export LD_LIBRARY_PATH=$OBIT_BASE_PATH/trunk/ObitSystem/Obit/lib:$LD_LIBRARY_PATH
# So we can import Obit.so and OTObit.py
export PYTHONPATH=$OBIT/python:$PYTHONPATH
# So OTObit.py can import AIPS.py
export PYTHONPATH=$OBIT_BASE_PATH/trunk/ObitSystem/ObitTalk/python:$PYTHONPATH

