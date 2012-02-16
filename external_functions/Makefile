#
# Top level Makefile for Ferret External Functions
#
# September 09 1998
# Jonathan Callahan
#
# ACM 1/2002  add fft directory back in: lsl_lowpass, inverse fcn.
# ACM 4/2006  link all functions that are distributed statically; the
#             directories below compile only example functions, functions
#             under development, or used locally by PMEL/TMAP (e.g.romea fcns)
 
#
# Site-specific defines
#
include ../site_specific.mk

#
# include platform specific definitions of:
#       CC, FC, RANLIB, CFLAGS, FFLAGS, PPLUS_FFLAGS
#
include ../platform_specific.mk.$(BUILDTYPE)

#
# Macro Definitions
#

DIRS = contributed examples fft statistics

#
# Targets
#

.PHONY : all
all: optimized

.PHONY : optimized
optimized:
	$(MAKE) "CFLAGS = $(CFLAGS) -O" "FFLAGS = $(FFLAGS) -O" build

.PHONY : debug
debug:
	$(MAKE) "CFLAGS = $(CFLAGS) -O0 -g" "FFLAGS = $(FFLAGS) -O0 -g" build

.PHONY: build
build:
	for i in $(DIRS) ; do \
            if ! $(MAKE) -C $$i ; then \
                exit 1 ; \
            fi ; \
        done

.PHONY: install
install:
	for i in $(DIRS) ; do \
            if ! $(MAKE) -C $$i install ; then \
                exit 1 ; \
            fi ; \
        done

.PHONY: clean
clean:
	for i in $(DIRS) ; do \
            if ! $(MAKE) -C $$i clean ; then \
                exit 1 ; \
            fi ; \
        done

#
# End of Makefile
#
