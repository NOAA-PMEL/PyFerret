#
# Makefile for building ferret
#

#
# Site-specific defines, including the definition of BUILDTYPE
#
include site_specific.mk

.PHONY : all
all : optimized

.PHONY : optimized
optimized :
	mkdir -p lib
	$(MAKE) xgks/Makefile
	$(MAKE) -C xgks
	$(MAKE) -C fer
	$(MAKE) -C external_functions
	$(MAKE) -C gksm2ps
	$(MAKE) -C bin/build_fonts/unix

.PHONY : debug
debug : 
	mkdir -p lib
	$(MAKE) xgks/Makefile
	$(MAKE) -C xgks
	$(MAKE) -C fer debug
	$(MAKE) -C external_functions debug
	$(MAKE) -C gksm2ps
	$(MAKE) -C bin/build_fonts/unix


## Configure xgks to create the Makefile if it does not exist
xgks/Makefile :
	( cd xgks; export BUILDTYPE=$(BUILDTYPE); ./configure )


## Clean all the directories
.PHONY : clean
clean :
	$(MAKE) -C bin/build_fonts/unix clean
	$(MAKE) -C gksm2ps clean
	$(MAKE) -C external_functions clean
	$(MAKE) -C fer clean
	rm -fr lib
	$(MAKE) xgksclean


## Thoroughly clean the xgks directory
.PHONY : xgksclean
xgksclean :
	$(MAKE) -C xgks clean
	$(MAKE) -C xgks distclean
	find xgks -name Makefile -exec rm -f {} \;
	rm -f xgks/port/master.mk
	rm -f xgks/port/udposix.h

##
