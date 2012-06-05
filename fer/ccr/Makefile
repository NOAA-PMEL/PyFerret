#
# include platform specific definitions of:
#       CC, FC, RANLIB, CFLAGS, FFLAGS, PPLUS_FFLAGS
#

include ../../site_specific.mk
include ../../platform_specific.mk.$(BUILDTYPE)
include ./SOURCE_FILES
include ./LIB_NAME

#
# Macros
#

OBJS = $(SRCS_C:.c=.o) $(SRCS_F:.F=.o)


#
# Targets
#

all:	$(LIB)

$(LIB): $(OBJS)
	$(AR) $(ARFLAGS) $@ $(OBJS)
	$(RANLIB) $@

#
# Special case for Perl interpreter
#
#perl_read.o : perl_read.c
#	cc -g -c perl_read.c `perl -MExtUtils::Embed -e ccopts`

clean:
	rm -f *.o $(LIB)

#
# End of Makefile
#
