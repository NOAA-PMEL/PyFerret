#
# Makefile (for the $(FER_LINK_DIR)/	ppl/addsxgks code)
#
# July 28 1997
# Jonathan Callahan (with some help from Joe Sirott)
#
# 10/97 *kob*  - pltform specific files will reside in ppl directory and be 
#		 links to actual files in fer directory
#
# include platform specific definitions of:
#       CC, FC, RANLIB, CFLAGS, FFLAGS, PPLUS_FFLAGS
#

include ./SOURCE_FILES
include ./LIB_NAME
#
# non-default suffix rules
#

.F.o:
	$(FC) $(PPLUS_FFLAGS) -c $<
 
# 10/97 *kob* - had to move this below to prevent the .F.o dependency from
#     	        being ignored on linux
include ../platform_specific_flags.mk.$(HOSTTYPE)

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
	rm -f core *.o


#
# End of Makefile
#
# DO NOT DELETE THIS LINE -- make depend depends on it.
