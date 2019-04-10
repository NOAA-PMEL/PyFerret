#
# Makefile for building ferret
#

#
# Site-specific defines, including the definition of BUILDTYPE
#
include site_specific.mk

ifeq ("$(BUILDTYPE)", "intel-mac")
	COPY_DYLIBS = - ./copy_dylibs.sh
else
	COPY_DYLIBS =
endif

.PHONY : all
all : optimized

.PHONY : optimized
optimized :
	mkdir -p lib
	$(COPY_DYLIBS)
	$(MAKE) xgks/Makefile
	$(MAKE) -C xgks
	$(MAKE) -C fer
	cp -f fer/ferret_c bin/ferret
	$(MAKE) "FER_DIR = $(DIR_PREFIX)" -C external_functions all
	rm -f bin/ferret
	$(MAKE) -C gksm2ps
	$(MAKE) -C bin/build_fonts/unix

.PHONY : debug
debug :
	mkdir -p lib
	$(COPY_DYLIBS)
	$(MAKE) xgks/Makefile
	$(MAKE) -C xgks
	$(MAKE) -C fer debug
	cp -f fer/ferret_c bin/ferret
	$(MAKE) "FER_DIR = $(DIR_PREFIX)" -C external_functions debug
	rm -f bin/ferret
	$(MAKE) -C gksm2ps
	$(MAKE) -C bin/build_fonts/unix

# Debug but also print all memory allocations, reallocations, and frees to 
# file "memorydebug.txt".  Initialize allocated memory with non-zero values. 
# Expect this to be a lot slower due to all the (intentionally inefficient 
# but safe) file operations.
.PHONY : memorydebug
memorydebug :
	mkdir -p lib
	$(COPY_DYLIBS)
	$(MAKE) xgks/Makefile
	$(MAKE) -C xgks
	$(MAKE) -C fer memorydebug
	cp -f fer/ferret_c bin/ferret
	$(MAKE) "FER_DIR = $(DIR_PREFIX)" -C external_functions debug
	rm -f bin/ferret
	$(MAKE) -C gksm2ps
	$(MAKE) -C bin/build_fonts/unix


## Configure xgks to create the Makefile if it does not exist
xgks/Makefile :
	( cd xgks; export BUILDTYPE=$(BUILDTYPE); ./configure )


## Clean all the directories
.PHONY : clean
clean :
	rm -f fer_executables.tar.gz
	rm -f fer_environment.tar.gz
	$(MAKE) -C bench clean
	$(MAKE) -C bin/build_fonts/unix clean
	$(MAKE) -C gksm2ps clean
	$(MAKE) -C external_functions clean
	$(MAKE) -C fer clean
	rm -f -R lib dylibs
	$(MAKE) xgksclean


## Thoroughly clean the xgks directory
.PHONY : xgksclean
xgksclean :
	$(MAKE) -C xgks clean
	$(MAKE) -C xgks distclean
	find xgks -name Makefile -exec rm -f {} \;
	rm -f xgks/port/master.mk
	rm -f xgks/port/udposix.h


## Install Ferret binaries, scripts, and other files into $(INSTALL_FER_DIR)
.PHONY : install
install : install_env install_exes

## Create the fer_environment.tar.gz files and then extract it into $(INSTALL_FER_DIR)
.PHONY :  install_env
install_env :
	rm -f fer_environment.tar.gz
	bin/make_environment_tar . . -y
	mkdir -p $(INSTALL_FER_DIR)
	mv -f fer_environment.tar.gz $(INSTALL_FER_DIR)
	( cd $(INSTALL_FER_DIR) ; tar xzf fer_environment.tar.gz )

## Create the fer_executables.tar.gz files and then extract it into $(INSTALL_FER_DIR)
.PHONY : install_exes
install_exes :
	rm -f fer_executables.tar.gz
	bin/make_executable_tar . . -y
	mkdir -p $(INSTALL_FER_DIR)
	mv -f fer_executables.tar.gz $(INSTALL_FER_DIR)
	( cd $(INSTALL_FER_DIR) ; tar xzf fer_executables.tar.gz )

## Compare results from executing the RUN_TESTS.sh test suite with those saved under bench/test_results
.PHONY : check
check :
	$(MAKE) -C bench check

.PHONY : check_noremote
check_noremote :
	$(MAKE) -C bench check_noremote

##
