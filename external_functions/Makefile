#
# Top level Makefile for Ferret External Functions
#
# September 09 1998
# Jonathan Callahan
#
 

#
# Macro Definitions
#

DIRS = examples sort v5d

#
# Targets
#

all:
	for i in $(DIRS); \
		do \
			(cd $$i; echo "making" $@ "in $$i..."; \
			$(MAKE) );\
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
