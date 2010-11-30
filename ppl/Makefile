#
# Makefile (for the $(FER_LINK_DIR)/ppl code)
#
# July 28 1997
# Jonathan Callahan (with some help from Joe Sirott)
#
# 10/97 *kob*  - pltform specific files will reside in ppl directory and be 
#		 links to actual files in fer directory
# v5.4 *kob* - add the removal of .f files - for linux


DIRS	= tmapadds complot ourlib plot \
	plotlib pplepic pplusr symlib

LIBS	= complot ourlib plot plotlib pplepic \
	pplusr symlib


#
# Targets
#

all:
	for i in $(DIRS) ; do \
            $(MAKE) -C $$i ; \
        done


libraries:
	for i in $(LIBS) ; do \
            $(MAKE) -C $$i ; \
        done


clean:
	for i in $(DIRS) ; do \
            $(MAKE) -C $$i clean ; \
        done


#
# End of Makefile
#
# DO NOT DELETE THIS LINE -- make depend depends on it.
