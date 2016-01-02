#
# Makefile for building and installing the pyferret module
# and the modules and libraries associated with it.
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
	$(MAKE) "CFLAGS = $(CFLAGS) -O" pymod_optimized
	if [ "$(BUILDTYPE)" != "intel-mac" ] ; then \
            if ! $(MAKE) "FFLAGS = $(FFLAGS) -O" -C $(DIR_PREFIX)/efmem ; then exit 1 ; fi ; \
            $(MAKE) "INSTALL_FER_DIR = $(DIR_PREFIX)/install" -C $(DIR_PREFIX)/external_functions optimized ; \
        fi
	$(MAKE) -C $(DIR_PREFIX)/bin/build_fonts/unix

.PHONY : debug
debug :
	mkdir -p $(DIR_PREFIX)/lib
	$(MAKE) -C $(DIR_PREFIX)/fer debug
	$(MAKE) -C $(DIR_PREFIX)/threddsBrowser
	$(MAKE) "CFLAGS = $(CFLAGS) -O0 -g" pymod_debug
	if [ "$(BUILDTYPE)" != "intel-mac" ] ; then \
            if ! $(MAKE) "FFLAGS = $(FFLAGS) -O0 -g" -C $(DIR_PREFIX)/efmem ; then exit 1 ; fi ; \
            $(MAKE) "INSTALL_FER_DIR = $(DIR_PREFIX)/install" -C $(DIR_PREFIX)/external_functions debug ; \
        fi
	$(MAKE) -C $(DIR_PREFIX)/bin/build_fonts/unix

## The following builds libpyferret.so, then installs that shared-object
## library and all the python scripts into $(DIR_PREFIX)/install.
.PHONY : pymod_optimized
pymod_optimized :
	rm -fr $(DIR_PREFIX)/build $(DIR_PREFIX)/install
	( cd $(DIR_PREFIX) ; \
	  export CAIRO_LIBDIR=$(CAIRO_LIBDIR) ; \
	  export PIXMAN_LIBDIR=$(PIXMAN_LIBDIR) ; \
	  export HDF5_LIBDIR=$(HDF5_LIBDIR) ; \
	  export NETCDF4_LIBDIR=$(NETCDF4_LIBDIR) ; \
	  export PYFERRET_VERSION=$(PYFERRET_VERSION) ; \
	  $(PYTHON_EXE) setup.py --quiet build )
	( cd $(DIR_PREFIX) ; \
	  export CAIRO_LIBDIR=$(CAIRO_LIBDIR) ; \
	  export PIXMAN_LIBDIR=$(PIXMAN_LIBDIR) ; \
	  export HDF5_LIBDIR=$(HDF5_LIBDIR) ; \
	  export NETCDF4_LIBDIR=$(NETCDF4_LIBDIR) ; \
	  export PYFERRET_VERSION=$(PYFERRET_VERSION) ; \
	  $(PYTHON_EXE) setup.py --quiet install -O2 --prefix=$(DIR_PREFIX)/install )

.PHONY : pymod_debug
pymod_debug :
	rm -fr $(DIR_PREFIX)/build $(DIR_PREFIX)/install
	( cd $(DIR_PREFIX) ; \
	  export CAIRO_LIBDIR=$(CAIRO_LIBDIR) ; \
	  export PIXMAN_LIBDIR=$(PIXMAN_LIBDIR) ; \
	  export HDF5_LIBDIR=$(HDF5_LIBDIR) ; \
	  export NETCDF4_LIBDIR=$(NETCDF4_LIBDIR) ; \
	  export PYFERRET_VERSION=$(PYFERRET_VERSION) ; \
	  $(PYTHON_EXE) setup.py --quiet build -g )
	( cd $(DIR_PREFIX) ; \
	  export CAIRO_LIBDIR=$(CAIRO_LIBDIR) ; \
	  export PIXMAN_LIBDIR=$(PIXMAN_LIBDIR) ; \
	  export HDF5_LIBDIR=$(HDF5_LIBDIR) ; \
	  export NETCDF4_LIBDIR=$(NETCDF4_LIBDIR) ; \
	  export PYFERRET_VERSION=$(PYFERRET_VERSION) ; \
	  $(PYTHON_EXE) setup.py --quiet install -O0 --prefix=$(DIR_PREFIX)/install )

## Remove everything that was built
.PHONY : clean
clean :
	$(MAKE) -C $(DIR_PREFIX)/bin/build_fonts/unix clean
	$(MAKE) -C $(DIR_PREFIX)/external_functions clean
	$(MAKE) -C $(DIR_PREFIX)/efmem clean
	rm -fr $(DIR_PREFIX)/install $(DIR_PREFIX)/build ferret.jnl*
	find $(DIR_PREFIX)/pviewmod -name '*.py[co]' -exec rm -f {} ';'
	find $(DIR_PREFIX)/pyfermod -name '*.py[co]' -exec rm -f {} ';'
	$(MAKE) -C $(DIR_PREFIX)/threddsBrowser clean
	$(MAKE) -C $(DIR_PREFIX)/fer clean
	rm -fr $(DIR_PREFIX)/lib

## Install Ferret binaries, scripts, and other files into $(INSTALL_FER_DIR)
.PHONY : install
install :
	rm -f pyferret-latest-local.tar.gz
	bin/make_dist_tar . latest local . -y
	mkdir -p $(INSTALL_FER_DIR)
	mv -f pyferret-latest-local.tar.gz $(INSTALL_FER_DIR)
	( cd $(INSTALL_FER_DIR) ; tar xz --strip-components=1 -f pyferret-latest-local.tar.gz )

## The following is for installing the updated threddsBrowser.jar, ferret_ef_meme_subsc.so,
## libpyferret.so, and PyFerret python scripts into $(INSTALL_FER_DIR)/lib without having 
## to use the distribution tar file.  Also copies all the PyFerret Fortran external function 
## to the $(INSTALL_FER_DIR)/ext_func/pylibs directory.
.PHONY : update
update :
	mkdir -p $(INSTALL_FER_DIR)/lib
	cp -f $(DIR_PREFIX)/threddsBrowser/threddsBrowser.jar $(INSTALL_FER_DIR)/lib
	cp -f $(DIR_PREFIX)/efmem/ferret_ef_mem_subsc.so $(INSTALL_FER_DIR)/lib
	find $(DIR_PREFIX)/external_functions -type f -perm -100 -name \*.so -exec cp {} $(INSTALL_FER_DIR)/ext_func/pylibs \;
	( cd $(DIR_PREFIX) ; \
	  export CAIRO_LIBDIR=$(CAIRO_LIBDIR) ; \
	  export PIXMAN_LIBDIR=$(PIXMAN_LIBDIR) ; \
	  export HDF5_LIBDIR=$(HDF5_LIBDIR) ; \
	  export NETCDF4_LIBDIR=$(NETCDF4_LIBDIR) ; \
	  export PYFERRET_VERSION=$(PYFERRET_VERSION) ; \
	  $(PYTHON_EXE) setup.py --quiet install -O2 --prefix=$(INSTALL_FER_DIR) )

##
