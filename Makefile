#
# Makefile for building and installing the pyferret module
# and the modules and libraries associated with it.
#

# Site-specific defines
include site_specific.mk

# Platform-specific defines
include platform_specific.mk.$(BUILDTYPE)

ifeq ("$(BUILDTYPE)", "intel-mac")
	COPY_DYLIBS = - ( cd $(DIR_PREFIX); ./copy_dylibs.sh )
else
	COPY_DYLIBS =
endif

.PHONY : all
all : optimized

.PHONY : optimized
optimized :
	mkdir -p $(DIR_PREFIX)/lib
	$(COPY_DYLIBS)
	$(MAKE) -C $(DIR_PREFIX)/fer optimized
	$(MAKE) pymod_optimized_build
	$(MAKE) pymod_optimized_install
	$(MAKE) externals_optimized
	$(MAKE) -C $(DIR_PREFIX)/bin/build_fonts/unix

.PHONY : debug
debug :
	mkdir -p $(DIR_PREFIX)/lib
	$(COPY_DYLIBS)
	$(MAKE) -C $(DIR_PREFIX)/fer debug
	$(MAKE) pymod_debug_build
	$(MAKE) pymod_debug_install
	$(MAKE) externals_debug
	$(MAKE) -C $(DIR_PREFIX)/bin/build_fonts/unix

## The definition of MEMORYDEBUG is observed by fer/special/FerMem_routines.c,
## which causes it to print (append) all memory allocations, reallocations,
## and frees to file "memorydebug.txt".  Initialize allocated memory with
## non-zero values.  Expect this to be a lot slower due to all the
## (intentionally inefficient but safe) file operations.
.PHONY : memorydebug
memorydebug :
	mkdir -p $(DIR_PREFIX)/lib
	$(COPY_DYLIBS)
	$(MAKE) -C $(DIR_PREFIX)/fer memorydebug
	$(MAKE) "CFLAGS = $(CFLAGS) -DMEMORYDEBUG" pymod_debug_build
	$(MAKE) "CFLAGS = $(CFLAGS) -DMEMORYDEBUG" pymod_debug_install
	$(MAKE) externals_debug
	$(MAKE) -C $(DIR_PREFIX)/bin/build_fonts/unix

## The following defines GRDELDEBUG used by fer/grdel, which causes it
## to create a grdeldebug.log file with all the graphics commands issued
.PHONY : grdeldebug
grdeldebug :
	mkdir -p $(DIR_PREFIX)/lib
	$(COPY_DYLIBS)
	$(MAKE) -C $(DIR_PREFIX)/fer grdeldebug
	$(MAKE) pymod_debug_build
	$(MAKE) pymod_debug_install
	$(MAKE) externals_debug
	$(MAKE) -C $(DIR_PREFIX)/bin/build_fonts/unix

## The following does an optimized build of libpyferret.so
.PHONY : pymod_optimized_build
pymod_optimized_build :
	rm -fr $(DIR_PREFIX)/build $(DIR_PREFIX)/install
	( cd $(DIR_PREFIX) ; \
	  export CC=$(CC) ; \
	  export CFLAGS="$(CFLAGS) -DNDEBUG -O" ; \
	  export BUILDTYPE=$(BUILDTYPE) ; \
	  export NETCDF_LIBDIR=$(NETCDF_LIBDIR) ; \
	  export HDF5_LIBDIR=$(HDF5_LIBDIR) ; \
	  export SZ_LIBDIR=$(SZ_LIBDIR) ; \
	  export CAIRO_LIBDIR=$(CAIRO_LIBDIR) ; \
	  export PIXMAN_LIBDIR=$(PIXMAN_LIBDIR) ; \
	  export PANGO_LIBDIR=$(PANGO_LIBDIR) ; \
	  export GLIB2_LIBDIR=$(GLIB2_LIBDIR) ; \
	  export GFORTRAN_LIB=$(GFORTRAN_LIB) ; \
	  export BIND_AND_HIDE_INTERNAL=$(BIND_AND_HIDE_INTERNAL) ; \
	  $(PYTHON_EXE) setup.py --quiet build )

## The following installs libpyferret.so and optimized
## versions of all the python scripts into $(DIR_PREFIX)/install.
.PHONY : pymod_optimized_install
pymod_optimized_install :
	rm -fr $(DIR_PREFIX)/install
	( cd $(DIR_PREFIX) ; \
	  export CC=$(CC) ; \
	  export CFLAGS="$(CFLAGS) -DNDEBUG -O" ; \
	  export BUILDTYPE=$(BUILDTYPE) ; \
	  export NETCDF_LIBDIR=$(NETCDF_LIBDIR) ; \
	  export HDF5_LIBDIR=$(HDF5_LIBDIR) ; \
	  export SZ_LIBDIR=$(SZ_LIBDIR) ; \
	  export CAIRO_LIBDIR=$(CAIRO_LIBDIR) ; \
	  export PIXMAN_LIBDIR=$(PIXMAN_LIBDIR) ; \
	  export PANGO_LIBDIR=$(PANGO_LIBDIR) ; \
	  export GLIB2_LIBDIR=$(GLIB2_LIBDIR) ; \
	  export GFORTRAN_LIB=$(GFORTRAN_LIB) ; \
	  export BIND_AND_HIDE_INTERNAL=$(BIND_AND_HIDE_INTERNAL) ; \
	  $(PYTHON_EXE) setup.py --quiet install -O2 --prefix=$(DIR_PREFIX)/install )

.PHONY : externals_optimized
externals_optimized :
	$(MAKE) "FER_DIR = $(DIR_PREFIX)/install" -C $(DIR_PREFIX)/external_functions optimized

## The following does a debug build of libpyferret.so
.PHONY : pymod_debug_build
pymod_debug_build :
	rm -fr $(DIR_PREFIX)/build $(DIR_PREFIX)/install
	( cd $(DIR_PREFIX) ; \
	  export CC=$(CC) ; \
	  export CFLAGS="$(CFLAGS) -UNDEBUG -O0 -g" ; \
	  export BUILDTYPE=$(BUILDTYPE) ; \
	  export NETCDF_LIBDIR=$(NETCDF_LIBDIR) ; \
	  export HDF5_LIBDIR=$(HDF5_LIBDIR) ; \
	  export SZ_LIBDIR=$(SZ_LIBDIR) ; \
	  export CAIRO_LIBDIR=$(CAIRO_LIBDIR) ; \
	  export PIXMAN_LIBDIR=$(PIXMAN_LIBDIR) ; \
	  export PANGO_LIBDIR=$(PANGO_LIBDIR) ; \
	  export GLIB2_LIBDIR=$(GLIB2_LIBDIR) ; \
	  export GFORTRAN_LIB=$(GFORTRAN_LIB) ; \
	  export BIND_AND_HIDE_INTERNAL=$(BIND_AND_HIDE_INTERNAL) ; \
	  $(PYTHON_EXE) setup.py build -g )

## The following installs libpyferret.so and unoptimized
## versions of all the python scripts into $(DIR_PREFIX)/install.
.PHONY : pymod_debug_install
pymod_debug_install :
	rm -fr $(DIR_PREFIX)/install
	( cd $(DIR_PREFIX) ; \
	  export CC=$(CC) ; \
	  export CFLAGS="$(CFLAGS) -UNDEBUG -O0 -g" ; \
	  export BUILDTYPE=$(BUILDTYPE) ; \
	  export NETCDF_LIBDIR=$(NETCDF_LIBDIR) ; \
	  export HDF5_LIBDIR=$(HDF5_LIBDIR) ; \
	  export SZ_LIBDIR=$(SZ_LIBDIR) ; \
	  export CAIRO_LIBDIR=$(CAIRO_LIBDIR) ; \
	  export PIXMAN_LIBDIR=$(PIXMAN_LIBDIR) ; \
	  export PANGO_LIBDIR=$(PANGO_LIBDIR) ; \
	  export GLIB2_LIBDIR=$(GLIB2_LIBDIR) ; \
	  export GFORTRAN_LIB=$(GFORTRAN_LIB) ; \
	  export BIND_AND_HIDE_INTERNAL=$(BIND_AND_HIDE_INTERNAL) ; \
	  $(PYTHON_EXE) setup.py --quiet install -O0 --prefix=$(DIR_PREFIX)/install )

.PHONY : externals_debug
externals_debug :
	$(MAKE) "FER_DIR = $(DIR_PREFIX)/install" -C $(DIR_PREFIX)/external_functions debug

## Remove everything that was built
.PHONY : clean
clean :
#	$(MAKE) -C $(DIR_PREFIX)/bench clean
	$(MAKE) -C $(DIR_PREFIX)/bin/build_fonts/unix clean
	$(MAKE) -C $(DIR_PREFIX)/external_functions clean
	rm -fr $(DIR_PREFIX)/install $(DIR_PREFIX)/build ferret.jnl*
	find $(DIR_PREFIX)/pviewmod -name '*.py[co]' -delete
	find $(DIR_PREFIX)/pyfermod -name '*.py[co]' -delete
	$(MAKE) -C $(DIR_PREFIX)/fer clean
	rm -fr $(DIR_PREFIX)/lib $(DIR_PREFIX)/dylibs

## Install Ferret binaries, scripts, and other files into $(INSTALL_FER_DIR)
.PHONY : install
install :
	rm -f pyferret-latest-local.tar.gz
	bin/make_dist_tar . latest local . -y
	mkdir -p $(INSTALL_FER_DIR)
	mv -f pyferret-latest-local.tar.gz $(INSTALL_FER_DIR)
	( cd $(INSTALL_FER_DIR) ; tar xz --strip-components=1 -f pyferret-latest-local.tar.gz )

## The following is for installing the updated ferret_ef_meme_subsc.so, libpyferret.so,
## and PyFerret python scripts into $(INSTALL_FER_DIR)/lib without having to use the
## distribution tar file.  Also copies all the PyFerret Fortran external function to
## the $(INSTALL_FER_DIR)/ext_func/pylibs directory.
.PHONY : update
update :
	mkdir -p $(INSTALL_FER_DIR)/lib
	find $(DIR_PREFIX)/external_functions -type f -perm -100 -name \*.so -exec cp {} $(INSTALL_FER_DIR)/ext_func/pylibs \;
	( cd $(DIR_PREFIX) ; \
	  export CC=$(CC) ; \
	  export CFLAGS="$(CFLAGS) -O" ; \
	  export BUILDTYPE=$(BUILDTYPE) ; \
	  export NETCDF_LIBDIR=$(NETCDF_LIBDIR) ; \
	  export HDF5_LIBDIR=$(HDF5_LIBDIR) ; \
	  export SZ_LIBDIR=$(SZ_LIBDIR) ; \
	  export CAIRO_LIBDIR=$(CAIRO_LIBDIR) ; \
	  export PIXMAN_LIBDIR=$(PIXMAN_LIBDIR) ; \
	  export PANGO_LIBDIR=$(PANGO_LIBDIR) ; \
	  export GLIB2_LIBDIR=$(GLIB2_LIBDIR) ; \
	  export GFORTRAN_LIB=$(GFORTRAN_LIB) ; \
	  export BIND_AND_HIDE_INTERNAL=$(BIND_AND_HIDE_INTERNAL) ; \
	  $(PYTHON_EXE) setup.py --quiet install -O2 --prefix=$(INSTALL_FER_DIR) )

## Compare results from executing the RUN_TESTS.sh test suite with those saved under bench/test_results
.PHONY : check
check :
	$(MAKE) -C $(DIR_PREFIX)/bench check

.PHONY : check_noremote
check_noremote :
	$(MAKE) -C $(DIR_PREFIX)/bench check_noremote

##
