#
# Top level Makefile for Ferret External Functions
#
# September 09 1998
# Jonathan Callahan
#
# ACM 1/2002  add fft directory back in: lsl_lowpass, inverse fcn.
#
 

#
# Macro Definitions
#

DIRS = cmprsi_by convolve date_and_time examples extrema fft sort v5d zaxr
DIRS_INSTALL = cmprsi_by convolve date_and_time  extrema fft sort v5d zaxr

#
# Targets
#

all:
	for i in $(DIRS); \
		do \
			(cd $$i; echo "making" $@ "in $$i..."; \
			$(MAKE) );\
		done

debug:
	for i in $(DIRS); \
		do \
			(cd $$i; echo "making debug" $@ "in $$i..."; \
			$(MAKE) debug);\
		done

install:
	for i in $(DIRS_INSTALL); \
		do \
			(cd $$i; echo "installing" $@ "in $$i..."; \
			$(MAKE) install );\
		done

clean:
	for i in $(DIRS); \
		do \
			(cd $$i; echo "cleaning" $@ "in $$i..."; \
			$(MAKE) clean);\
		done

#
# End of Makefile
#
