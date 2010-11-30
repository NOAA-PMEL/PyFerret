#
# Makefile (for the FERRET/fer/special code)
#
# August 11 1997
# Jonathan Callahan (with some help from Joe Sirott)
#
# 10/97 *kob*  - pltform specific files will reside in ppl directory and be 
#		 links to actual files in fer directory

#
# include platform specific definitions of:
#       CC, FC, RANLIB, CFLAGS, FFLAGS, PPLUS_FFLAGS
#

include ../../platform_specific_flags.mk.$(HOSTTYPE)
include ./SOURCE_FILES



#
# Macros
#

OBJS = $(SRCS_C:.c=.o) $(SRCS_F:.F=.o)


#
# Targets
#

all:	$(OBJS)


clean:
	rm -f core *.o *.a *.f


#
# End of Makefile
#
