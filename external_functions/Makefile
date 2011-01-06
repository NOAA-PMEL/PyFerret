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
# Macro Definitions
#

DIRS = contributed examples fft ncedit statistics las

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
	for i in $(DIRS); \
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
