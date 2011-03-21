# Site-dependent definitions included in Makefiles

# Full path name of the directory containing this file (the ferret root directory).
# Do not use $(shell pwd) since this is included in Makefiles in other directories.
# DIR_PREFIX	:= $(HOME)/pyferret_32dev
DIR_PREFIX	:= $(HOME)/pyferret_64dev

# Python 2.x executable to invoke for build and install.
# PYTHON_EXE	:= python2.4
PYTHON_EXE	:= python2.6

# Flags for specifying the installation directory for "$(PYTHON_EXE) setup.py install"
# PYTHON_INSTALL_FLAGS	:= --prefix=$(HOME)/.local
PYTHON_INSTALL_FLAGS	:= --user

# Java 1.6 jdk home directory ( $(JAVA_HOME)/bin/javac is called to build threddsBrowser.jar ).
JAVA_HOME	:= /usr/java/latest

#
