#
# Makefile (for the $(FER_LINK_DIR)/	ppl/addsxgks code)
#
# July 28 1997
# Jonathan Callahan (with some help from Joe Sirott)
#

#
# include platform specific definitions of:
#       CC, FC, RANLIB, CFLAGS, FFLAGS, PPLUS_FFLAGS
#

include ../../platform_specific_flags.mk.$(HOSTTYPE)
include ./SOURCE_FILES
#
# non-default suffix rules
#

.F.o:
	$(FC) $(PPLUS_FFLAGS) -c $<
 

#
# Macros
#

OBJS = $(SRCS_C:.c=.o) $(SRCS_F:.F=.o)


#
# Targets
#

all:	$(OBJS)


depend:
	makedepend $(PPLUS_CFLAGS) $(SRCS_C)
 

clean:
	rm -f core *.o


#
# End of Makefile
#
# DO NOT DELETE THIS LINE -- make depend depends on it.
