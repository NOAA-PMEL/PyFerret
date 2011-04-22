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
optimized : optimizedbuild install

.PHONY : debug
debug : debugbuild install

.PHONY : optimizedbuild
optimizedbuild :
	mkdir -p $(DIR_PREFIX)/lib
	cp $(READLINE_DIR)/lib/libreadline.a $(READLINE_DIR)/lib/libhistory.a $(DIR_PREFIX)/lib
	$(MAKE) -C $(DIR_PREFIX)/fer optimized
	$(MAKE) pymod
	$(MAKE) -C $(DIR_PREFIX)/external_functions optimized

.PHONY : debugbuild
debugbuild : 
	mkdir -p $(DIR_PREFIX)/lib
	cp $(READLINE_DIR)/lib/libreadline.a $(READLINE_DIR)/lib/libhistory.a $(DIR_PREFIX)/lib
	$(MAKE) -C $(DIR_PREFIX)/fer debug
	$(MAKE) "CFLAGS += -O0 -g" pymod
	$(MAKE) -C $(DIR_PREFIX)/external_functions debug

.PHONY : pymod
pymod :
	rm -fr $(DIR_PREFIX)/build
	cd $(DIR_PREFIX) ; export HDF5_DIR=$(HDF5_DIR) ; export NETCDF_DIR=$(NETCDF_DIR) ; $(PYTHON_EXE) setup.py build

.PHONY : install
install :
	cp -f $(DIR_PREFIX)/fer/threddsBrowser/threddsBrowser.jar $(FER_LIBS)
	cd $(DIR_PREFIX) ; export HDF5_DIR=$(HDF5_DIR) ; export NETCDF_DIR=$(NETCDF_DIR) ; $(PYTHON_EXE) setup.py install $(PYTHON_INSTALL_FLAGS)
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
