#!/bin/bash

# For future reference, the following are the environment variables
# created by setup.sh, in turn created by calling InstallObit.sh
# This may be instructive in illustrating the concrete environment
# variables created below

# OBIT=/src/Obit/trunk/ObitSystem/Obit; export OBIT
# OBITSD=/src/Obit/trunk/ObitSystem/ObitSD; export OBITSD
# PYTHONPATH="/src/Obit/trunk/ObitSystem/ObitSD/python:/src/Obit/trunk/ObitSystem/Obit/python:/src/Obit/trunk/opt/share/obittalk/python/"; export PYTHONPATH
# PATH="/src/Obit/trunk/bin:$PATH"; export PATH
# OBITINSTALL=/src/Obit/trunk; export OBITINSTALL
# PLPLOT_DRV_DIR=/src/Obit/trunk/other/lib/plplot5.8.0/drivers; export PLPLOT_DRV_DIR

export OBIT_BASE_PATH=/usr/local/Obit
export OBIT=$OBIT_BASE_PATH/ObitSystem/Obit
export OBITINSTALL=$OBIT_BASE_PATH
export OBITSD=$OBIT_BASE_PATH/ObitSystem/ObitSD
export OBIT_EXEC=$OBIT

# This makes no sense yet, because we don't copy anything to /bin
#export PATH=$OBIT_BASE_PATH/bin:$PATH

# Add OBIT's bin directory
export PATH=$OBIT_BASE_PATH/ObitSystem/Obit/bin:$PATH

# So we can import libObit.so
export LD_LIBRARY_PATH=$OBIT_BASE_PATH/ObitSystem/Obit/lib:$LD_LIBRARY_PATH
# So we can import Obit.so and OTObit.py
export PYTHONPATH=$OBIT_BASE_PATH/ObitSystem/Obit/python:$PYTHONPATH
export PYTHONPATH=$OBIT_BASE_PATH/ObitSystem/ObitSD/python:$PYTHONPATH
# So OTObit.py can import AIPS.py
export PYTHONPATH=/usr/local/share/obittalk/python:$PYTHONPATH

