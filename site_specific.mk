# Site-dependent definitions included in Makefiles

# Full path name of the directory containing this file (the ferret root directory).
# Do not use $(shell pwd) since this is included in Makefiles in other directories.
DIR_PREFIX	:= $(H)/pyferret_32dev
# DIR_PREFIX	:= $(H)/pyferret_64dev

# Flags for python 2.x and numpy include directories.
PYINC_FLAGS	:= -I/usr/include/python2.4 -I/usr/lib/python2.4/site-packages/numpy/core/include
# PYINC_FLAGS	:= -I/usr/local/include/python2.6 -I/usr/local/lib/python2.6/site-packages/numpy/core/include

# Flags for the python 2.x library.
PYLIB_FLAGS	:= -lpython2.4
# PYLIB_FLAGS	:= -lpython2.6

# Python 2.x executable to invoke for build and install.
PYTHON_EXE	:= python2.4
# PYTHON_EXE	:= python2.6

# Flags for specifying the installation directory for "$(PYTHON_EXE) setup.py install"
PYTHON_INSTALL_FLAGS	:= --prefix=$(H)/.local
# PYTHON_INSTALL_FLAGS	:= --user

# Java 1.6 jdk home directory ( $(JAVA_HOME)/bin/javac is called to build threddsBrowser.jar ).
JAVA_HOME	:= /usr/java/latest

#
