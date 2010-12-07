#
# Makefile for the static library used for interfacing with Python external functions
#

#
# Site-specific defines
#
include ../site_specific.mk

#
# Local defines
#
FERINC_FLAGS	:= -I$(DIR_PREFIX)/fer/ef_utility
FERINC_FILES	:= $(DIR_PREFIX)/fer/ef_utility/EF_Util.h

.PHONY : all
all : libpyefcn.a

.PHONY : lib
lib : libpyefcn.a

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
	rm -f *.o *.a

# build rule for C file objects in this directory
CSRCOBJS := $(patsubst %.c, %.o, $(wildcard *.c))
$(CSRCOBJS): %.o: %.c pyefcn.h $(FERINC_FILES) Makefile
	$(CC) $(CFLAGS) $(FERINC_FLAGS) $(PYINC_FLAGS) -fPIC -c $< -o $@

libpyefcn.a : $(CSRCOBJS) Makefile
	$(AR) $(ARFLAGS) libpyefcn.a $(CSRCOBJS)

#
