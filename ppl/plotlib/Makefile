#
# Makefile (for the $(FER_LINK_DIR)/	ppl/addsxgks code)
#
# July 28 1997
# Jonathan Callahan (with some help from Joe Sirott)
#
# 10/97 *kob*  - pltform specific files will reside in ppl directory and be 
#		 links to actual files in fer directory
# 2/03 *kob* g77 port - move include for platform specifics file
#                       to correct place and fix f77 macro reference
# 2/03 *kob* - remove unneedd .F dependency - get it from platform specifics 
#              flag	
#
# include platform specific definitions of:
#       CC, FC, RANLIB, CFLAGS, FFLAGS, PPLUS_FFLAGS
#

include ../platform_specific_flags.mk.$(HOSTTYPE)
include ./SOURCE_FILES
include ./LIB_NAME
#
# non-default suffix rules
#

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


depend:
	makedepend $(PPLUS_CFLAGS) $(SRCS_C)
 

clean:
	rm -f core *.o *.f


#
# End of Makefile
#
# DO NOT DELETE THIS LINE -- make depend depends on it.
