#
# Makefile for building the ferret shared-object library (libferret.so), 
# the ferret interface shared-object library for python external functions (libpyefcn.so),
# and the pyferret module with its shared-object library (_pyferret.so).
#

DIR_PREFIX	:= $(HOME)/pyferret_32dev
PYINC_FLAGS	:= -I/usr/local/include/python2.6 -I/usr/local/lib/python2.6/site-packages/numpy/core/include
PYLIB_FLAGS	:= -lpython2.6


.PHONY : all
all : optimized

.PHONY : optimized
optimized : optimizedbuild install

.PHONY : debug
debug : debugbuild install

.PHONY : optimizedbuild
optimizedbuild :
	$(MAKE) -C $(DIR_PREFIX)/fer optimizedbuild
	$(MAKE) -C $(DIR_PREFIX)/ferlib optimizedlib
	$(MAKE) -C $(DIR_PREFIX)/pyefcn optimizedlib
	$(MAKE) pymod

.PHONY : debugbuild
debugbuild : 
	$(MAKE) -C $(DIR_PREFIX)/fer debugbuild
	$(MAKE) -C $(DIR_PREFIX)/ferlib debuglib
	$(MAKE) -C $(DIR_PREFIX)/pyefcn debuglib
	$(MAKE) "CFLAGS += -O0 -g" pymod

.PHONY : pymod
pymod :
	python2.6 setup.py build

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
	cp -f $(DIR_PREFIX)/pyefcn/libpyefcn.so $(FER_LIBS)
ifeq ( $(USER), "root" )
	python2.6 setup.py install --skip-build
else
	python2.6 setup.py install --skip-build --user
endif
endif

.PHONY : clean
clean :
	rm -fr build ferret.jnl*
	$(MAKE) -C $(DIR_PREFIX)/pyefcn clean
	$(MAKE) -C $(DIR_PREFIX)/ferlib clean
	@echo ""
	@echo "    NOTE: Only the (pyferret) build, ferlib, and pyefcn directories were cleaned."
	@echo "          Use target 'distclean' to also clean the fer, fmt, and ppl directories."
	@echo ""

.PHONY : distclean
distclean :
	rm -fr build ferret.jnl*
	$(MAKE) -C $(DIR_PREFIX)/pyefcn clean
	$(MAKE) -C $(DIR_PREFIX)/ferlib clean
	$(MAKE) -C $(DIR_PREFIX)/fer clean
	@echo ""
	@echo "    NOTE: Only the (pyferret) build, ferlib, pyefcn, fer, fmt, and ppl directories were cleaned."
	@echo "          Other directories (in particular, xgks) were not changed."
	@echo ""

#
