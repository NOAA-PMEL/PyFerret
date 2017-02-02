#
# Makefile for the graphics delegate library
#

include ../../site_specific.mk
include ../../platform_specific.mk.$(BUILDTYPE)

LIBNAME = ../../lib/libgrdel.a
CHDRS = $(wildcard *.h) ../cferbind/cferbind.h
CSRCS = $(wildcard *.c)
COBJS = $(CSRCS:.c=.o)
FSRCS = $(wildcard *.F)
FOBJS = $(FSRCS:.F=.o)

# CFLAGS already modified to include debug or optimization flags
# Just add the additional warning flags to CFLAGS
.PHONY : all
all : 
	$(MAKE) "CFLAGS = $(CFLAGS) -Wall -Werror" $(LIBNAME)

$(LIBNAME) : $(COBJS) $(FOBJS)
	$(AR) $(ARFLAGS) $(LIBNAME) $(COBJS) $(FOBJS)
	$(RANLIB) $(LIBNAME)

$(COBJS) : $(CHDRS)

.PHONY : clean
clean :
	rm -f *.o $(LIBNAME)

#
# End of Makefile
#

