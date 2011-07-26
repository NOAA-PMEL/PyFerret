#
# Makefile for the ppl libraries
#

DIRS	= tmapadds complot ourlib plot \
	  plotlib pplepic pplusr symlib

#
# Targets
#

all:
	for i in $(DIRS) ; do \
            if ! $(MAKE) -C $$i ; then \
                exit 1 ; \
            fi ; \
        done


clean:
	for i in $(DIRS) ; do \
            $(MAKE) -C $$i clean ; \
        done


#
# End of Makefile
#
