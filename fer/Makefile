#
# Simple makefile for launching other makes to build all the ferret libraries
#

#
# Site-specific defines
#
include ../site_specific.mk

#
# include platform specific definitions of:
#       CC, FC, RANLIB, CFLAGS, FFLAGS, PPLUS_FFLAGS
#
include ../platform_specific.mk.$(BUILDTYPE)

PLATFORM := $(shell uname -rs)
VERSION_INFO := (PyFerret beta 5)
DATE := $(shell /bin/date +%D)

#
# Macros
#

DIRS = ccr ctx dat doo efi efn ef_utility fmt gnl ino mem  \
       ocn plt pyf rpn special stk utl xeq threddsBrowser

LIBS = ccr ctx dat doo efi efn ef_utility fmt gnl ino mem \
       ocn plt pyf rpn stk utl xeq

#
# Targets
#


.PHONY : all
all: optimized


.PHONY : optimized
optimized:
	$(MAKE) "CFLAGS = $(CFLAGS) -O -DNDEBUG" \
		"FFLAGS = $(FFLAGS) -O -DNDEBUG" build


.PHONY : debug
debug:
	$(MAKE) "CFLAGS = $(CFLAGS) -O0 -g" \
		"FFLAGS = $(FFLAGS) -O0 -g" build


.PHONY : build
build: update special_code ppl_libraries fmt_libraries fer_libraries threddsBrowser


.PHONY : update
update:
	rm -f special/xmake_date_data.[Ffo]
	sed "s&xxxxxxxx&$(DATE)&" < special/xmake_date_data.template > special/xmake_date_data.F
	rm -f special/xrevision_type_data.[Ffo]
	sed -e 's&mmmmmmmm&$(VERSION_INFO)&' -e 's&(/&(&' -e 's&/)&)&' < special/xrevision_type_data.template > special/xrevision_type_data.F 
	rm -f special/xplatform_type_data.[Ffo]
	sed -e 's&mmmmmmmm&$(PLATFORM)&' -e 's&(/&(&' -e 's&/)&)&' < special/xplatform_type_data.template > special/xplatform_type_data.F 


.PHONY : special_code
special_code: update
	# the XGKS library needs to be build prior to building pyferret
	cp ../xgks/src/lib/libxgks.a ../lib
	$(MAKE) -C special all


.PHONY : ppl_libraries
ppl_libraries:
	$(MAKE) -C ../ppl


.PHONY : fmt_libraries
fmt_libraries:
	$(MAKE) -C ../fmt/src


.PHONY : fer_libraries
fer_libraries: update
	for i in $(LIBS) ; do \
            if ! $(MAKE) -C $$i ; then \
                exit 1 ; \
            fi ; \
        done


.PHONY : threddsBrowser
threddsBrowser:
	$(MAKE) -C threddsBrowser all


.PHONY : clean
clean: ppl_clean fmt_clean fer_clean
	rm -f ../lib/*


.PHONY : ppl_clean
ppl_clean:
	$(MAKE) -C ../ppl clean


.PHONY : fmt_clean
fmt_clean:
	$(MAKE) -C ../fmt/src clean


.PHONY : fer_clean
fer_clean:
	for i in $(DIRS) ; do \
            $(MAKE) -C $$i clean ; \
        done
	rm -f special/xmake_date_data.F special/xrevision_type_data.F special/xplatform_type_data.F

