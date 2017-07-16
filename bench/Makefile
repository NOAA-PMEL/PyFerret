#
# Makefile for the PyFerret bench subdirectory
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
	( SED_EXTREGEX_FLAG="$(SED_EXTREGEX_FLAG)"; export SED_EXTREGEX_FLAG; ./RUN_TESTS.sh `which pyferret` "$(PYFER_EXTERNAL_FUNCTIONS)" )
	@echo 'Differences in the generated bench PDF and PNG files versus those in bench/test_results'
	-@for img in *.pdf *.png ; \
	do \
	    if diff -q $$img test_results/$$img ; then \
	        echo "Files $$img and test_results/$$img are identical"; \
	    else \
	        compare $$img test_results/$$img "$$img"_diff.png ; \
	    fi \
	done
	-@( datetag=`date +'%d%h%yAT'|tr '[A-Z]' '[a-z]'` ; \
	    diff all_$${datetag}*_ncdump test_results/pyferret_run_tests_ncdump 1> pyferret_run_tests_ncdump.diff 2>&1 ; \
	    diff all_$${datetag}*_err test_results/pyferret_run_tests_err 1> pyferret_run_tests_err.diff 2>&1 ; \
	    diff all_$${datetag}*_log test_results/pyferret_run_tests_log 1> pyferret_run_tests_log.diff 2>&1 \
	)
	@echo ''
	@echo 'Inspect the *_diff.png files for actual differences in the images'
	@echo 'Inspect the pyferret_run_tests_*.diff files for actual differences in the output'

.PHONY: clean
clean:
	rm -f all_*_ncdump all_*_err all_*_log all_*_ncdump_orig all_*_err_orig all_*_log_orig cleanups.sed
	rm -f *.ps* *.eps* *.png* *.pdf* *.diff* ferret.jnl*
	rm -f `cat TRASH_FILES`
	rm -f all_ncdump.out fort.41 make_stream_file
	rm -fr subdir tmp

make_stream_file: make_stream_file.F
	$(FC) $(FFLAGS) make_stream_file.F -o make_stream_file

#
# End of Makefile
#
