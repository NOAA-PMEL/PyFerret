#
# This makefile assumes that make has already been run in the xgks, fer, and pyefcn subdirectories
# to build xgks/src/lib/libxgks.a, pyefcn/libpyefcn.a, the required static libraries in the lib subdirectory,
# and the required object files under the fer/ef_utility, fer/special, and ppl/tmapadds subdirectories.
#

#
# Site-specific defines
#
include ../site_specific.mk

#
# Local defines
#
FERINC_FLAGS	:= -I$(DIR_PREFIX)/fer/common -I$(DIR_PREFIX)/fmt/cmn -I$(DIR_PREFIX)/ppl/tmap_inc
FERINC_FILES	:= $(wildcard $(DIR_PREFIX)/fer/common/*) \
		   $(wildcard $(DIR_PREFIX)/fmt/cmn/*) \
		   $(wildcard $(DIR_PREFIX)/ppl/tmap_inc/*)

FEROBJS_EFU	:= $(wildcard $(DIR_PREFIX)/fer/ef_utility/*.o)
FEROBJS_SPL	:= $(DIR_PREFIX)/fer/special/fakes3.o $(DIR_PREFIX)/fer/special/ferret_dispatch.o \
		   $(DIR_PREFIX)/fer/special/ferret_query_f.o $(DIR_PREFIX)/fer/special/gui_fakes.o \
		   $(DIR_PREFIX)/fer/special/linux_routines.o $(wildcard $(DIR_PREFIX)/fer/special/x*_data.o)
FEROBJS_TMA	:= $(wildcard $(DIR_PREFIX)/ppl/tmapadds/*.o)
FEROBJS_FILES	:= $(FEROBJS_EFU) $(FEROBJS_SPL) $(FEROBJS_TMA)

FERLIBS_FILES	:= $(wildcard $(DIR_PREFIX)/lib/lib*.a) \
		   $(DIR_PREFIX)/pyefcn/libpyefcn.a $(DIR_PREFIX)/xgks/src/lib/libxgks.a 

.PHONY : all
all : ferret

.PHONY : lib
lib : libferret.so

.PHONY : optimized
optimized :
	$(MAKE) "CFLAGS += -O2" "FFLAGS += -O2" all

.PHONY : debug
debug :
	$(MAKE) "CFLAGS += -O0 -g" "FFLAGS += -O0 -g" all

.PHONY : optimizedlib
optimizedlib :
	$(MAKE) "CFLAGS += -O2" "FFLAGS += -O2" lib

.PHONY : debuglib
debuglib :
	$(MAKE) "CFLAGS += -O0 -g" "FFLAGS += -O0 -g" lib

.PHONY : clean
clean :
	rm -f *.o *.so ferret ferret.jnl*

# build rule for Fortran file objects in this directory
FSRCOBJS := $(patsubst %.f, %.o, $(wildcard *.f))
$(FSRCOBJS): %.o: %.f $(FERINC_FILES) Makefile
	$(F77) $(FFLAGS) $(FERINC_FLAGS) -ffixed-line-length-132 -fPIC -c $< -o $@

# build rule for C file objects in this directory
ALLCOBJS := $(patsubst %.c, %.o, $(wildcard *.c))
$(ALLCOBJS): %.o: %.c ferret_lib.h $(FERINC_FILES) Makefile
	$(CC) $(CFLAGS) $(FERINC_FLAGS) $(PYINC_FLAGS) -fPIC -c $< -o $@

# don't include main in the library
CSRCOBJS := $(filter-out fermain_so.o, $(ALLCOBJS))

libferret.so : $(CSRCOBJS) $(FSRCOBJS) $(FEROBJS_FILES) $(FERLIBS_FILES) Makefile
	$(F77) $(FFLAGS) $(LDFLAGS) -shared \
	    $(CSRCOBJS) $(FSRCOBJS) $(FEROBJS_FILES) \
	    -Xlinker --whole-archive \
	    $(FERLIBS_FILES) \
	    -Xlinker --no-whole-archive \
	    $(PYLIB_FLAGS) \
	    -lnetcdff -lnetcdf -lhdf5_hl -lhdf5 \
            -lreadline -lhistory -lncurses -lX11 -lcurl -lz -ldl \
	    -Xlinker --no-undefined \
	    -o libferret.so

ferret : fermain_so.o libferret.so Makefile
	$(CC) $(CFLAGS) fermain_so.o libferret.so -o ferret

#
