#
# Makefile (for the FERRET/fer/pyf code)
#

#
# Site-specific defines
#
include ../../site_specific.mk

#
# include platform specific definitions of:
#       CC, FC, RANLIB, CFLAGS, FFLAGS, PPLUS_FFLAGS
#
include ../../platform_specific.mk.$(HOSTTYPE)

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
	rm -f *.o *.f $(LIB)

#
# End of Makefile
#
