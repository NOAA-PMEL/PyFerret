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

include platform_specific_flags.mk.$(HOSTTYPE)


#
# Macros
#

DIRS	= tmapadds complot ourlib plot \
	plotlib pplepic pplp_only pplusr symlib

LIBS	= complot ourlib plot plotlib pplepic \
	pplusr symlib

CLEANDIRS = $(DIRS)

DEPDIRS = addsxgks modsxgks pplp_only


#
# Targets
#

all:
	for i in $(DIRS) ; \
        do \
        (cd $$i ; echo "making" $@ "in $$i..."; \
        $(MAKE) "MFLAGS=$(MFLAGS)" "CC=$(CC)" "CFLAGS=$(CFLAGS)" \
                "FFLAGS=$(FFLAGS)" "PPLUS_FFLAGS=$(PPLUS_FFLAGS)"); \
        done


libraries:
	for i in $(LIBS) ; \
        do \
        (cd $$i ; echo "making" $@ "in $$i..."; \
        $(MAKE) "MFLAGS=$(MFLAGS)" "CC=$(CC)" "CFLAGS=$(CFLAGS)" \
                "FFLAGS=$(FFLAGS)" "PPLUS_FFLAGS=$(PPLUS_FFLAGS)"); \
        done


clean:
	for i in $(CLEANDIRS) ; \
        do \
        (cd $$i ; echo "making" $@ "in $$i..."; rm -f *.f; $(MAKE) clean); \
        done

depend:
	for i in $(DEPDIRS) ; \
        do \
        (cd $$i ; echo "making" $@ "in $$i..."; \
        $(MAKE) "MFLAGS=$(MFLAGS)" "CC=$(CC)" "CFLAGS=$(CFLAGS)" \
                "FFLAGS=$(FFLAGS)" "PPLUS_FFLAGS=$(PPLUS_FFLAGS)" $@); \
        done


#
# End of Makefile
#
# DO NOT DELETE THIS LINE -- make depend depends on it.
