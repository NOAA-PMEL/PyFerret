#
# Makefile for the graphics delegate library
#

include ../../site_specific.mk
include ../../platform_specific.mk.$(BUILDTYPE)

LIBNAME = ../../lib/libgrdel.a
CHDRS = $(wildcard *.h)
CSRCS = $(wildcard *.c)
COBJS = $(CSRCS:.c=.o)
FSRCS = $(wildcard *.F)
FOBJS = $(FSRCS:.F=.o)

.PHONY : all
all : optimized

.PHONY : optimized
optimized :
	$(MAKE) "CFLAGS = $(CFLAGS) -DNDEBUG -O" \
		"FFLAGS = $(FFLAGS) -DNDEBUG -O" $(LIBNAME)

.PHONY : debug
debug :
	$(MAKE) "CFLAGS = $(CFLAGS) -DVERBOSEDEBUG -O0 -g" \
		"FFLAGS = $(FFLAGS) -O0 -g" $(LIBNAME)

$(LIBNAME) : $(COBJS) $(FOBJS)
	$(AR) $(ARFLAGS) $(LIBNAME) $(COBJS) $(FOBJS)
	$(RANLIB) $(LIBNAME)

$(COBJS) : $(CHDRS)

.PHONY : clean
clean :
	rm -f *.o *.f $(LIBNAME)

#
# End of Makefile
#

