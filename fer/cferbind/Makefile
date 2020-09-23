#
# Makefile for Ferret bindings written in C
#

include ../../site_specific.mk
include ../../platform_specific.mk.$(BUILDTYPE)

LIBNAME = ../../lib/libcferbind.a
CHDRS = $(wildcard *.h) ../common/ferret.h ../grdel/grdel.h
CSRCS = $(wildcard *.c)
COBJS = $(CSRCS:.c=.o)

# CFLAGS already modified to include debug or optimization flags
# Just add the additional warning flags to CFLAGS
.PHONY : all
all :
	# $(MAKE) "CFLAGS = $(CFLAGS) -Wall -Wno-pointer-to-int-cast -Werror" $(LIBNAME)
	$(MAKE) "CFLAGS = $(CFLAGS) -Wall -Wno-pointer-to-int-cast" $(LIBNAME)

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
