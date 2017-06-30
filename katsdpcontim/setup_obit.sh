#!/bin/sh
cd /src/Obit
patch -p0 < /obit.patch

cd /src/Obit/trunk/ObitSystem/Obit
./configure --prefix=/usr --without-plplot --without-wvr
make -j 8
