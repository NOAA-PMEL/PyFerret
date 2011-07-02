#
# Makefile for building and installing the ferret shared-object library 
# (libferret.so), the pyferret module with its shared-object library 
# (_pyferret.so), and the ferret.py script.
#

#
# Site-specific defines
#
include site_specific.mk

.PHONY : all
all : optimized

.PHONY : optimized
optimized :
	mkdir -p $(DIR_PREFIX)/lib
	cp $(READLINE_DIR)/lib/libreadline.a $(READLINE_DIR)/lib/libhistory.a $(DIR_PREFIX)/lib
	$(MAKE) -C $(DIR_PREFIX)/fer optimized
	$(MAKE) "CFLAGS += -O" pymod
	$(MAKE) -C $(DIR_PREFIX)/external_functions optimized

.PHONY : debug
debug : 
	mkdir -p $(DIR_PREFIX)/lib
	cp $(READLINE_DIR)/lib/libreadline.a $(READLINE_DIR)/lib/libhistory.a $(DIR_PREFIX)/lib
	$(MAKE) -C $(DIR_PREFIX)/fer debug
	$(MAKE) "CFLAGS += -O0 -g" pymod
	$(MAKE) -C $(DIR_PREFIX)/external_functions debug

## The following builds _pyferret.so, then installs that shared-object library and all the
## python scripts into $(DIR_PREFIX)/pyferret_install.  This install directory can then be
## used for the <pyferret_install_dir> argument to make_executables_tar.
.PHONY : pymod
pymod :
	rm -fr $(DIR_PREFIX)/build $(DIR_PREFIX)/pyferret_install
	cd $(DIR_PREFIX) ; export HDF5_DIR=$(HDF5_DIR) ; export NETCDF_DIR=$(NETCDF_DIR) ; $(PYTHON_EXE) setup.py install --prefix=$(DIR_PREFIX)/pyferret_install

## The following is for installing the updated threddsBrowser.jar into $(FER_LIBS) and 
## the updated _pyferret.so and python scripts into $(FER_DIR)/lib without having to go
## through the make_executables_tar script.
.PHONY : install
install :
	mkdir -p $(FER_LIBS)
	cp -f $(DIR_PREFIX)/fer/threddsBrowser/threddsBrowser.jar $(FER_LIBS)
	cd $(DIR_PREFIX) ; export HDF5_DIR=$(HDF5_DIR) ; export NETCDF_DIR=$(NETCDF_DIR) ; $(PYTHON_EXE) setup.py install --prefix=$(FER_DIR)
	# $(MAKE) -C $(DIR_PREFIX)/external_functions install
	@echo "***** NOTE: external functions not installed *****"

.PHONY : clean
clean :
	$(MAKE) -C $(DIR_PREFIX)/external_functions clean
	rm -fr $(DIR_PREFIX)/build ferret.jnl*
	$(MAKE) -C $(DIR_PREFIX)/fer clean
	rm -fr $(DIR_PREFIX)/lib
	@echo ""
	@echo "    NOTE: Only the build, external_functions, fer, fmt, ppl,"
	@echo "          and lib directories were cleaned.  Other directories"
	@echo "          (in particular, xgks) were not changed."
	@echo ""

#
