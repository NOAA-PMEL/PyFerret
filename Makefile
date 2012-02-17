#
# Makefile for building and installing the ferret shared-object library
# (libferret.so), the pyferret module with its shared-object library
# (_pyferret.so), and the ferret.py script.
#

# Site-specific defines
include site_specific.mk

# Platform-specific defines
include platform_specific.mk.$(BUILDTYPE)

.PHONY : all
all : optimized

.PHONY : optimized
optimized :
	mkdir -p $(DIR_PREFIX)/lib
	$(MAKE) -C $(DIR_PREFIX)/fer optimized
	$(MAKE) -C $(DIR_PREFIX)/threddsBrowser
	$(MAKE) "CFLAGS = $(CFLAGS) -O2" pymod_optimized
	$(MAKE) -C $(DIR_PREFIX)/external_functions optimized
	$(MAKE) -C $(DIR_PREFIX)/bin/build_fonts/unix

.PHONY : debug
debug :
	mkdir -p $(DIR_PREFIX)/lib
	$(MAKE) -C $(DIR_PREFIX)/fer debug
	$(MAKE) -C $(DIR_PREFIX)/threddsBrowser
	$(MAKE) "CFLAGS = $(CFLAGS) -O0 -g" pymod_debug
	$(MAKE) -C $(DIR_PREFIX)/external_functions debug
	$(MAKE) -C $(DIR_PREFIX)/bin/build_fonts/unix

## The following builds _pyferret.so, then installs that shared-object library and all the
## python scripts into $(DIR_PREFIX)/pyferret_install.  This install directory can then be
## used for the <pyferret_install_dir> argument to make_executables_tar.
.PHONY : pymod_optimized
pymod_optimized :
	rm -fr $(DIR_PREFIX)/build $(DIR_PREFIX)/pyferret_install
	( cd $(DIR_PREFIX) ; \
	  export HDF5_LIBDIR=$(HDF5_LIBDIR) ; \
	  export NETCDF4_LIBDIR=$(NETCDF4_LIBDIR) ; \
	  export PYFERRET_VERSION=$(PYFERRET_VERSION) ; \
	  $(PYTHON_EXE) setup.py build )
	( cd $(DIR_PREFIX) ; \
	  export HDF5_LIBDIR=$(HDF5_LIBDIR) ; \
	  export NETCDF4_LIBDIR=$(NETCDF4_LIBDIR) ; \
	  export PYFERRET_VERSION=$(PYFERRET_VERSION) ; \
	  $(PYTHON_EXE) setup.py install -O2 --prefix=$(DIR_PREFIX)/pyferret_install )

.PHONY : pymod_debug
pymod_debug :
	rm -fr $(DIR_PREFIX)/build $(DIR_PREFIX)/pyferret_install
	( cd $(DIR_PREFIX) ; \
	  export HDF5_LIBDIR=$(HDF5_LIBDIR) ; \
	  export NETCDF4_LIBDIR=$(NETCDF4_LIBDIR) ; \
	  export PYFERRET_VERSION=$(PYFERRET_VERSION) ; \
	  $(PYTHON_EXE) setup.py build -g )
	( cd $(DIR_PREFIX) ; \
	  export HDF5_LIBDIR=$(HDF5_LIBDIR) ; \
	  export NETCDF4_LIBDIR=$(NETCDF4_LIBDIR) ; \
	  export PYFERRET_VERSION=$(PYFERRET_VERSION) ; \
	  $(PYTHON_EXE) setup.py install -O0 --prefix=$(DIR_PREFIX)/pyferret_install )

## Remove everything that was built
.PHONY : clean
clean :
	rm -fr fer_executables.tar.gz
	rm -fr fer_environment.tar.gz
	$(MAKE) -C $(DIR_PREFIX)/bin/build_fonts/unix clean
	$(MAKE) -C $(DIR_PREFIX)/external_functions clean
	rm -fr $(DIR_PREFIX)/pyferret_install $(DIR_PREFIX)/build ferret.jnl*
	find $(DIR_PREFIX)/pviewmod -name '*.py[co]' -exec rm -f {} ';'
	find $(DIR_PREFIX)/pyfermod -name '*.py[co]' -exec rm -f {} ';'
	$(MAKE) -C $(DIR_PREFIX)/threddsBrowser clean
	$(MAKE) -C $(DIR_PREFIX)/fer clean
	rm -fr $(DIR_PREFIX)/lib

## Install Ferret binaries, scripts, and other files into $(INSTALL_FER_DIR)
.PHONY : install
install : install_env install_exes

## Create the fer_environment.tar.gz files and then extract it into $(INSTALL_FER_DIR)
.PHONY :  install_env
install_env :
	rm -f fer_environment.tar.gz
	bin/make_environment_tar . . -y
	mkdir -p $(INSTALL_FER_DIR)
	mv -f fer_environment.tar.gz $(INSTALL_FER_DIR)
	( cd $(INSTALL_FER_DIR) ; tar xvzf fer_environment.tar.gz )

## Create the fer_executables.tar.gz files and then extract it into $(INSTALL_FER_DIR)
.PHONY : install_exes
install_exes :
	rm -f fer_executables.tar.gz
	bin/make_executable_tar . . -y
	mkdir -p $(INSTALL_FER_DIR)
	mv -f fer_executables.tar.gz $(INSTALL_FER_DIR)
	( cd $(INSTALL_FER_DIR) ; tar xvzf fer_executables.tar.gz )
	cp -f threddsBrowser/toolsUI/toolsUI-4.1.jar $(INSTALL_FER_DIR)/lib/

## The following is for installing the updated threddsBrowser.jar, _pyferret.so, 
## and python scripts into $(INSTALL_FER_DIR)/lib without having to go
## through the make_executables_tar script.
.PHONY : update
update :
	mkdir -p $(INSTALL_FER_DIR)/lib
	cp -f $(DIR_PREFIX)/threddsBrowser/threddsBrowser.jar $(INSTALL_FER_DIR)/lib
	( cd $(DIR_PREFIX) ; \
	  export HDF5_LIBDIR=$(HDF5_LIBDIR) ; \
	  export NETCDF4_LIBDIR=$(NETCDF4_LIBDIR) ; \
	  export PYFERRET_VERSION=$(PYFERRET_VERSION) ; \
	  $(PYTHON_EXE) setup.py install -O2 --prefix=$(INSTALL_FER_DIR) )

##
