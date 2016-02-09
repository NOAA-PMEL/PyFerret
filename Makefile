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
	$(MAKE) -C threddsBrowser
	$(MAKE) -C external_functions
	$(MAKE) -C gksm2ps
	$(MAKE) -C bin/build_fonts/unix

.PHONY : beta
beta :
	mkdir -p lib
	$(MAKE) xgks/Makefile
	$(MAKE) -C xgks
	$(MAKE) -C fer beta
	$(MAKE) -C threddsBrowser
	$(MAKE) -C external_functions
	$(MAKE) -C gksm2ps
	$(MAKE) -C bin/build_fonts/unix

.PHONY : debug
debug :
	mkdir -p lib
	$(MAKE) xgks/Makefile
	$(MAKE) -C xgks
	$(MAKE) -C fer debug
	$(MAKE) -C threddsBrowser
	$(MAKE) -C external_functions debug
	$(MAKE) -C gksm2ps
	$(MAKE) -C bin/build_fonts/unix


## Configure xgks to create the Makefile if it does not exist
xgks/Makefile :
	( cd xgks; export BUILDTYPE=$(BUILDTYPE); ./configure )


## Clean all the directories
.PHONY : clean
clean :
	rm -fr fer_executables.tar.gz
	rm -fr fer_environment.tar.gz
	$(MAKE) -C bin/build_fonts/unix clean
	$(MAKE) -C gksm2ps clean
	$(MAKE) -C external_functions clean
	$(MAKE) -C threddsBrowser clean
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

##
