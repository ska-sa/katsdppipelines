#!/bin/bash
export OBIT_BASE_PATH=/src/Obit
export OBIT=$OBIT_BASE_PATH/trunk/ObitSystem/Obit
export OBITINSTALL=$OBIT_BASE_PATH/trunk
export OBITSD=$OBIT_BASE_PATH/trunk/ObitSystem/ObitSD
export OBIT_EXEC=$OBIT

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
make clean
make -j 8

# This makes no sense yet, because we don't create anything in /bin
#export PATH=$OBIT_BASE_PATH/trunk/bin:$PATH
export PATH=$OBIT_BASE_PATH/trunk/ObitSystem/Obit/bin:$PATH

# So we can import libObit.so
export LD_LIBRARY_PATH=$OBIT_BASE_PATH/trunk/ObitSystem/Obit/lib:$LD_LIBRARY_PATH
# So we can import Obit.so and OTObit.py
export PYTHONPATH=$OBIT_BASE_PATH/trunk/ObitSystem/Obit/python:$PYTHONPATH
export PYTHONPATH=$OBIT_BASE_PATH/trunk/ObitSystem/ObitSD/python:$PYTHONPATH
# So OTObit.py can import AIPS.py
export PYTHONPATH=$OBIT_BASE_PATH/trunk/ObitSystem/ObitTalk/python:$PYTHONPATH

# OBIT=/src/Obit/trunk/ObitSystem/Obit; export OBIT
# OBITSD=/src/Obit/trunk/ObitSystem/ObitSD; export OBITSD
# PYTHONPATH="/src/Obit/trunk/ObitSystem/ObitSD/python:/src/Obit/trunk/ObitSystem/Obit/python:/src/Obit/trunk/opt/share/obittalk/python/"; export PYTHONPATH
# PATH="/src/Obit/trunk/bin:$PATH"; export PATH
# OBITINSTALL=/src/Obit/trunk; export OBITINSTALL
# PLPLOT_DRV_DIR=/src/Obit/trunk/other/lib/plplot5.8.0/drivers; export PLPLOT_DRV_DIR
