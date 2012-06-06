#
# Site-specific defines
#
include ../../site_specific.mk

#
# include platform specific definitions of:
#       CC, FC, RANLIB, CFLAGS, FFLAGS, PPLUS_FFLAGS
#
include ../../platform_specific.mk.$(BUILDTYPE)

include ./SOURCE_FILES
include ./LIB_NAME

#
# Macros
#

OBJS = $(SRCS_F:.F=.o)


#
# Targets
#

all:	$(LIB)

$(LIB): $(OBJS)
	$(AR) $(ARFLAGS) $@ $(OBJS)
	$(RANLIB) $@

clean:
	rm -f *.o $(LIB)

#
# End of Makefile
#
