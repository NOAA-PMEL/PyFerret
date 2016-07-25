#
# Makefile for the PyFerret source bench subdirectory
#

# include platform specific macro definitions for make_stream_file
include ../site_specific.mk
include ../platform_specific.mk.$(BUILDTYPE)

## Execute the RUN_TESTS.sh test suite
.PHONY: run_tests
run_tests:
	@if [ `which pyferret` = "" -o "$(FER_DIR)" = "" ]; then \
	    echo "*** PyFerret must be installed and the ferret_paths script must have sourced for these tests ***" ; \
	    exit 1 ; \
	fi
	./RUN_TESTS.sh `which pyferret` "$(PYFER_EXTERNAL_FUNCTIONS)" "RUN_TESTS.sh test results for PyFerret"
	@echo 'Differences in the generated bench PDF and PNG files versus against those in test_results'
	-@for img in *.pdf *.png ; \
	do \
	    diff -q $$img test_results/$$img ; \
	done
	@echo ''
	@echo 'Inspect the bench/all_*_ncdump, bench/all_*_err, and bench/all_*_log files just created '
	@echo 'against bench/test_results/pyferret_run_tests_ncdump, bench/test_results/pyferret_run_test_err, '
	@echo 'and bench/test_results/pyferret_run_tests_log'

.PHONY: clean
clean:
	rm -f all_*_ncdump all_*_err all_*_log
	rm -f *.plt* *.gif* *.ps* *.png* *.pdf* ferret.jnl*
	rm -f `cat TRASH_FILES`
	rm -f all_ncdump.out fort.41 make_stream_file
	rm -fr subdir tmp

make_stream_file: make_stream_file.F
	$(FC) $(FFLAGS) make_stream_file.F -o make_stream_file

#
# End of Makefile
#
