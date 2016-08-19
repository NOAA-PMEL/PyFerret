#
# Makefile for Ferret bindings written in C
#

include ../../site_specific.mk
include ../../platform_specific.mk.$(BUILDTYPE)

LIBNAME = ../../lib/libcferbind.a
CHDRS = $(wildcard *.h) ../grdel/grdel.h ../ccr/utf8str.h
CSRCS = $(wildcard *.c)
COBJS = $(CSRCS:.c=.o)

.PHONY : all
all : optimized

# The -I$(CAIRO_DIR)/include/cairo is only needed for older cairo's (1.2) own include files

.PHONY : optimized
optimized :
	$(MAKE) "CFLAGS = $(CFLAGS) -I$(CAIRO_DIR)/include/cairo -I../ccr -Wall -Wno-pointer-to-int-cast -Werror -DNDEBUG -O" $(LIBNAME)

.PHONY : debug
debug :
	$(MAKE) "CFLAGS = $(CFLAGS) -I$(CAIRO_DIR)/include/cairo -I../ccr -Wall -Wno-pointer-to-int-cast -Werror -O0 -g" $(LIBNAME)

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

