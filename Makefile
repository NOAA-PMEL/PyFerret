#
# Makefile for building the ferret shared-object library (libferret.so), 
# and the pyferret module with its shared-object library (_pyferret.so).
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
	$(MAKE) -C $(DIR_PREFIX)/fer optimizedbuild
	$(MAKE) -C $(DIR_PREFIX)/pyefcn optimizedlib
	$(MAKE) -C $(DIR_PREFIX)/ferlib optimizedlib
	$(MAKE) pymod

.PHONY : debugbuild
debugbuild : 
	mkdir -p $(DIR_PREFIX)/lib
	$(MAKE) -C $(DIR_PREFIX)/fer debugbuild
	$(MAKE) -C $(DIR_PREFIX)/pyefcn debuglib
	$(MAKE) -C $(DIR_PREFIX)/ferlib debuglib
	$(MAKE) "CFLAGS += -O0 -g" pymod

.PHONY : pymod
pymod :
	$(PYTHON_EXE) setup.py build

.PHONY : install
install :
ifeq ( $(strip $(FER_LIBS)), )
	@echo ""
	@echo " ERROR: environment variable FER_LIBS is not defined"
	@echo "        installation unsuccessful"
	@echo ""
else
	cp -f $(DIR_PREFIX)/fer/threddsBrowser/threddsBrowser.jar $(FER_LIBS)
	cp -f $(DIR_PREFIX)/ferlib/libferret.so $(FER_LIBS)
	$(PYTHON_EXE) setup.py install $(PYTHON_INSTALL_FLAGS)
endif

.PHONY : clean
clean :
	rm -fr $(DIR_PREFIX)/build ferret.jnl*
	$(MAKE) -C $(DIR_PREFIX)/pyefcn clean
	$(MAKE) -C $(DIR_PREFIX)/ferlib clean
	@echo ""
	@echo "    NOTE: Only the (pyferret) build, ferlib, and pyefcn directories were cleaned."
	@echo "          Use target 'distclean' to also clean the fer, fmt, ppl, and lib directories."
	@echo ""

.PHONY : distclean
distclean :
	rm -fr $(DIR_PREFIX)/build ferret.jnl*
	$(MAKE) -C $(DIR_PREFIX)/pyefcn clean
	$(MAKE) -C $(DIR_PREFIX)/ferlib clean
	$(MAKE) -C $(DIR_PREFIX)/fer clean
	rm -fr $(DIR_PREFIX)/lib
	@echo ""
	@echo "    NOTE: Only the (pyferret) build, ferlib, pyefcn, lib, fer, fmt, ppl, and lib directories were cleaned."
	@echo "          Other directories (in particular, xgks) were not changed."
	@echo ""

#
