#
# Makefile (for the FERRET/fer/ccr code)
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

#CFLAGS += -I../gui
#
# Macros
#

#OBJS = $(SRCS_F:.F=.o)
OBJS = $(SRCS_C:.c=.o) $(SRCS_F:.F=.o)


#
# Targets
#

all:	$(LIB)


$(LIB): $(OBJS)
	$(AR) $(ARFLAGS) $@ $(OBJS)
	$(RANLIB) $@

ferret_cmn:
	ln -s ../common ferret_cmn

#
# Special case for Perl interpreter
#
#perl_read.o : perl_read.c
#	cc -g -c perl_read.c `perl -MExtUtils::Embed -e ccopts`

clean:
	rm -f core *.o *.a *.f

#
# End of Makefile
#
