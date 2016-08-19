export NETCDF4_LIBDIR=/usr/local/lib NETCDF4_DIR=/usr/local
export PYTHON_EXE=python2.7 FER_DIR=/opt/PyFerret
export HOSTTYPE=intel-mac-ifort DIR_PREFIX=/usr/local/src/pyferret_src
export ARCHFLAGS="-arch x86_64"
make && make install
