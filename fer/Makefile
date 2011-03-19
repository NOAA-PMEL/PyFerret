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
include ../platform_specific.mk.$(HOSTTYPE)

DATE = $(shell /bin/date +%D)

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
	$(MAKE) "CFLAGS += -O2" "FFLAGS += -O2" build


.PHONY : debug
debug:
	$(MAKE) "CFLAGS += -O0 -g" "FFLAGS += -O0 -g" build


.PHONY : build
build: update special_code ppl_libraries fmt_libraries fer_libraries threddsBrowser


.PHONY : update
update:
	rm -f special/xmake_date_data.[Ffo]
	sed "s&xxxxxxxx&$(DATE)&" < special/xmake_date_data.template > special/xmake_date_data.F
	rm -f special/xrevision_type_data.[Ffo]
	sed -e 's&mmmmmmmm&&' -e 's&(/&(&' -e 's&/)&)&' < special/xrevision_type_data.template > special/xrevision_type_data.F 
	rm -f special/xplatform_type_data.[Ffo]
	sed -e 's&mmmmmmmm&$(PLATFORM)&' -e 's&(/&(&' -e 's&/)&)&' < special/xplatform_type_data.template > special/xplatform_type_data.F 


.PHONY : special_code
special_code: update
	$(MAKE) -C special all


.PHONY : ppl_libraries
ppl_libraries:
	$(MAKE) -C ../ppl all


.PHONY : fmt_libraries
fmt_libraries:
	$(MAKE) -C ../fmt/src


.PHONY : fer_libraries
fer_libraries: update
	for i in $(LIBS) ; do \
            $(MAKE) -C $$i ; \
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

