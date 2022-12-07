#
# Top level Makefile for Ferret External Functions
#
# September 09 1998
# Jonathan Callahan
# 
# External functions are used to develop new functions. If they
# are seen to have likely wider use, they are moved to the main
# code base under fer/efi with minor changes to the way include
# files are specified. 
#
# Those that remain here to be built as part of the distribution are
# in the examples/ directory, and a few others including the string 
# functions in stringfcns/which could be incorporated into the main 
# code and documented.
#
#
# ACM 1/2002  add fft directory back in: lsl_lowpass, inverse fcn.
# ACM 4/2006  link all functions that are distributed statically; the
#             directories below compile only example functions, functions
#             under development, or used locally by PMEL/TMAP (e.g.romea fcns)
# ACM 12/2022 Remove decimate from the directories to build: these functions
#             are linked into the PyFerret program
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

# DIRS = ef_utility
DIRS = ef_utility examples fft statistics stringfcns
MOREDIRS = cmprsi_by extrema decimate featurecollections gridding tax_times

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

.PHONY: buildmore
buildmore:
	for i in $(DIRS) $(MOREDIRS) ; do \
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

.PHONY: installmore
installmore:
	for i in $(DIRS) $(MOREDIRS) ; do \
            if ! $(MAKE) -C $$i install ; then \
                exit 1 ; \
            fi ; \
        done

.PHONY: clean
clean:
	for i in $(DIRS) $(MOREDIRS) ; do \
            if ! $(MAKE) -C $$i clean ; then \
                exit 1 ; \
            fi ; \
        done

#
# End of Makefile
#
