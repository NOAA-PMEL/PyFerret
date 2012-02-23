#
# Makefile (for the $(FER_LINK_DIR)/ppl code)
#
# July 28 1997
# Jonathan Callahan (with some help from Joe Sirott)
#
# 10/97 *kob*  - pltform specific files will reside in ppl directory and be 
#		 links to actual files in fer directory
# v5.4 *kob* - add the removal of .f files - for linux

#
# include platform specific definitions of:
#       CC, FC, RANLIB, CFLAGS, FFLAGS, PPLUS_FFLAGS
#

include ../site_specific.mk
include ../platform_specific.mk.$(BUILDTYPE)


#
# Macros
#

DIRS	= tmapadds complot ourlib plot \
	plotlib pplepic pplp_only pplusr symlib

LIBS	= complot ourlib plot plotlib pplepic \
	pplusr symlib

CLEANDIRS = $(DIRS)

DEPDIRS = addsxgks modsxgks pplp_only

# change the standard rules for .F -> .o builds so $(PPLUS_FFLAGS) is used
%.o : %.F
	rm -f $*.f
	$(CPP) -P -traditional $(CPP_FLAGS) $(<F) | sed -e 's/de    /de /g' | sed -e 's/de         /de /g' > $*.f
	$(F77) $(PPLUS_FFLAGS) -c $*.f	 


#
# Targets
#

all:
	for i in $(DIRS) ; \
        do \
            if ! $(MAKE) -C $$i ; then \
                exit 1 ; \
            fi ; \
        done


libraries:
	for i in $(LIBS) ; \
        do \
            if ! $(MAKE) -C $$i ; then \
                exit 1 ; \
            fi ; \
        done


clean:
	for i in $(CLEANDIRS) ; \
        do \
            $(MAKE) -C $$i clean ; \
        done

depend:
	for i in $(DEPDIRS) ; \
        do \
            $(MAKE) -C $$i depend ; \
        done


#
# End of Makefile
#
# DO NOT DELETE THIS LINE -- make depend depends on it.
