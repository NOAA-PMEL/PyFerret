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
DATE := $(shell /bin/date +%D)

#
# Macros
#

DIRS = agg ccr cferbind ctx dat doo efi efn ef_utility fmt gnl grdel \
       ino mem plt pyf rpn stk utl xeq special

LIBS = agg ccr cferbind ctx dat doo efi efn ef_utility fmt gnl grdel \
       ino mem plt pyf rpn stk utl xeq 

#
# Targets
#

.PHONY : all
all: optimized


.PHONY : optimized
optimized:
	$(MAKE) "CFLAGS = $(CFLAGS) $(OPT_FLAGS)" \
		"FFLAGS = $(FFLAGS) $(OPT_FLAGS)" \
		"PPLUS_FFLAGS = $(PPLUS_FFLAGS) $(PPLUS_OPT_FLAGS)" \
		"VERSION_INFO = (optimized)" \
		build

.PHONY : beta
beta:
	$(MAKE) "CFLAGS = $(CFLAGS) $(OPT_FLAGS)" \
		"FFLAGS = $(FFLAGS) $(OPT_FLAGS)" \
		"PPLUS_FFLAGS = $(PPLUS_FFLAGS) $(PPLUS_OPT_FLAGS)" \
		"VERSION_INFO = (beta/optim)" \
		build

.PHONY : debug
debug:
	$(MAKE) "CFLAGS = $(CFLAGS) $(DBG_FLAGS)"  \
		"FFLAGS = $(FFLAGS) $(DBG_FLAGS) -fbounds-check" \
		"PPLUS_FFLAGS = $(PPLUS_FFLAGS) $(DBG_FLAGS)" \
		"VERSION_INFO = (beta/debug)" \
		build

## The definition of MEMORYDEBUG is observed by fer/special/FerMem_routines.c, 
## which causes it to print (append) all memory allocations, reallocations, 
## and frees to file "memorydebug.txt".  Initialize allocated memory with 
## non-zero values.  Expect this to be a lot slower due to all the 
## (intentionally inefficient but safe) file operations.
.PHONY : memorydebug
memorydebug:
	$(MAKE) "CFLAGS = $(CFLAGS) $(DBG_FLAGS) -DMEMORYDEBUG"  \
		"FFLAGS = $(FFLAGS) $(DBG_FLAGS) -fbounds-check" \
		"PPLUS_FFLAGS = $(PPLUS_FFLAGS) $(DBG_FLAGS)" \
		"VERSION_INFO = (beta/debug)" \
		build


## The definition of GRDELDEBUG is observed by fer/grdel, which causes it
## to create a grdeldebug.log file with all the graphics commands issued
.PHONY : grdeldebug
grdeldebug:
	$(MAKE) "CFLAGS = $(CFLAGS) $(DBG_FLAGS) -DGRDELDEBUG"  \
		"FFLAGS = $(FFLAGS) $(DBG_FLAGS) -fbounds-check" \
		"PPLUS_FFLAGS = $(PPLUS_FFLAGS) $(DBG_FLAGS)" \
		"VERSION_INFO = (beta/debug)" \
		build

.PHONY: profiling
profiling:
	$(MAKE) "CFLAGS = $(CFLAGS) $(PROF_FLAGS) "  \
		"FFLAGS = $(FFLAGS) $(PROF_FLAGS) -fbounds-check" \
		"PPLUS_FFLAGS = $(PPLUS_FFLAGS) $(PROF_FLAGS)" \
		"LDFLAGS = $(LDFLAGS) $(PROF_FLAGS)" \
		"VERSION_INFO = (profiling)" \
		build

.PHONY : build
build: update special_code ppl_libraries fmt_libraries fer_libraries


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

#
# End of Makefile
#
