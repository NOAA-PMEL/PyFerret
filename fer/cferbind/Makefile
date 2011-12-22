#
# Makefile for Ferret bindings written in C
#

include ../../site_specific.mk
include ../../platform_specific.mk.$(BUILDTYPE)

LIBNAME = ../../lib/libcferbind.a
CHDRS = $(wildcard *.h)
CSRCS = $(wildcard *.c)
COBJS = $(CSRCS:.c=.o)

.PHONY : all
all : optimized

.PHONY : optimized
optimized :
	$(MAKE) "CFLAGS = $(CFLAGS) -DNDEBUG -O" $(LIBNAME)

.PHONY : debug
debug :
	$(MAKE) "CFLAGS = $(CFLAGS) -O0 -g" $(LIBNAME)

$(LIBNAME) : $(COBJS)
	$(AR) $(ARFLAGS) $(LIBNAME) $(COBJS)
	$(RANLIB) $(LIBNAME)

$(COBJS) : $(CHDRS)

.PHONY : clean
clean :
	rm -f *.o $(LIBNAME)

#
# End of Makefile
#

