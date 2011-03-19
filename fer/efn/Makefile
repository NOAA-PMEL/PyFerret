#
# Makefile (for the FERRET/fer/efn code)
#
# August 11 1997
# Jonathan Callahan (with some help from Joe Sirott)
# 10/97 *kob*  - pltform specific files will reside in ppl directory and be 
#		 links to actual files in fer directory
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
