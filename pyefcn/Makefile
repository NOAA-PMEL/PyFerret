#
# Makefile for the shared-object library used for interfacing with Python external functions
# This build assumes the ferret shared object library is already built.
#

#
# These are assigned only if they are not already defined
#
DIR_PREFIX	?= $(HOME)/pyferret_32dev
PYINC_FLAGS	?= -I/usr/local/include/python2.6 -I/usr/local/lib/python2.6/site-packages/numpy/core/include
PYLIB_FLAGS	?= -lpython2.6

#
# Local defines
#
FERINC_FLAGS	:= -I$(DIR_PREFIX)/fer/ef_utility
FERINC_FILES	:= $(DIR_PREFIX)/fer/ef_utility/EF_Util.h
FERLIB_FLAGS	:= -L$(DIR_PREFIX)/ferlib -lferret

.PHONY : all
all : libpyefcn.so

.PHONY : lib
lib : libpyefcn.so

.PHONY : optimized
optimized :
	$(MAKE) "CFLAGS += -O2" all

.PHONY : debug
debug :
	$(MAKE) "CFLAGS += -O0 -g" all

.PHONY : optimizedlib
optimizedlib :
	$(MAKE) "CFLAGS += -O2" lib

.PHONY : debuglib
debuglib :
	$(MAKE) "CFLAGS += -O0 -g" lib

.PHONY : clean
clean :
	rm -f *.o *.so

# build rule for C file objects in this directory
CSRCOBJS := $(patsubst %.c, %.o, $(wildcard *.c))
$(CSRCOBJS): %.o: %.c pyefcn.h $(FERINC_FILES) Makefile
	$(CC) $(CFLAGS) $(FERINC_FLAGS) $(PYINC_FLAGS) -fPIC -c $< -o $@

libpyefcn.so : $(CSRCOBJS) Makefile
	$(CC) $(CFLAGS) -shared $(CSRCOBJS) $(FERLIB_FLAGS) $(PYLIB_FLAGS) -o libpyefcn.so

#
