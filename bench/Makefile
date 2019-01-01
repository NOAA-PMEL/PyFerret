#
# Makefile for the PyFerret bench subdirectory
#

# include platform specific macro definitions for make_stream_file
include ../site_specific.mk
include ../platform_specific.mk.$(BUILDTYPE)

## Run all the benchmark tests, including those requiring remote data
## (environment variable SKIP_REMOTE is empty)
.PHONY: check
check:
	@$(MAKE) "SKIP_REMOTE=" run_tests
	@$(MAKE) "SKIP_REMOTE=" compare_plots
	@$(MAKE) "SKIP_REMOTE=" compare_logs

## Run all the benchmark tests except those requiring remote data
## (environment variable SKIP_REMOTE is not empty)
.PHONY: check_noremote
check_noremote:
	@$(MAKE) "SKIP_REMOTE=XXXX" run_tests
	@$(MAKE) "SKIP_REMOTE=XXXX" compare_plots
	@$(MAKE) "SKIP_REMOTE=XXXX" compare_logs

## Execute the RUN_TESTS.sh test suite
## If the environment variable SKIP_REMOTE is not empty, skips tests requiring remote data
.PHONY: run_tests
run_tests:
	@( \
	    ferret_exe=`which pyferret` ;\
	    if [ $$ferret_exe = "" -o "$(FER_DIR)" = "" ] ; then \
	        echo "*** PyFerret must be installed and the ferret_paths script must have sourced for these tests ***" ;\
	        exit 1 ;\
	    fi ;\
	    echo "Using $$ferret_exe" ;\
	    SED_EXTREGEX_FLAG="$(SED_EXTREGEX_FLAG)" ;\
	    export SED_EXTREGEX_FLAG ;\
	    ./RUN_TESTS.sh $$ferret_exe "$(PYFER_EXTERNAL_FUNCTIONS)" \
	)

## Compare the plots from executing the RUN_TESTS.sh test suite with those saved.
## If the environment variable SKIP_REMOTE is empty, compares to those under "test_result";
## otherwise compares to those under "test_results_noremote"
.PHONY: compare_plots
compare_plots:
	-@( \
	    if [ -z "$(SKIP_REMOTE)" ] ; then \
	        results_dir="test_results" ;\
	    else \
	        results_dir="test_results_noremote" ;\
	    fi ;\
	    echo "Comparing PNG and PDF files to those in bench/$$results_dir" ;\
	    for img in `basename $$results_dir/*.png $$results_dir/*.pdf *.png *.pdf | sort -u` ; do \
	        if [ ! -f $$img ] ; then \
	            echo "*** File $$img is missing ***" ;\
	        elif [ ! -f $$results_dir/$$img ] ; then \
	            echo "*** File $$results_dir/$$img is missing ***" ;\
	        elif diff -q $$img $$results_dir/$$img ; then \
	            echo "Files $$img and $$results_dir/$$img are identical" ;\
	        else \
	            compare $$img $$results_dir/$$img $${img}_diff.png ; \
	        fi ;\
	    done \
	)
	@echo ''
	@echo 'Inspect the *_diff.png files for actual differences in the images'
	@echo ''

## Compare the log, err, and ncdump results from executing the RUN_TESTS.sh test suite.
## If the environment variable SKIP_REMOTE is empty, compares to those under "test_result";
## otherwise compares to those under "test_results_noremote"
.PHONY: compare_logs
compare_logs:
	-@( \
	    if [ -z "$(SKIP_REMOTE)" ] ; then \
	        results_dir="test_results" ;\
	    else \
	        results_dir="test_results_noremote" ;\
	    fi ;\
	    datetag=`date +'%d%h%yAT'|tr '[A-Z]' '[a-z]'` ;\
	    logfile=`/bin/ls -1 all_$${datetag}*_log` ;\
	    if [ -z "$$logfile" ] ; then \
	        echo "*** No run_tests results log file from today found ***" ;\
	        exit 1 ;\
	    fi ;\
	    if [ `echo "$${logfile[@]}" | wc -l` -ne 1 ] ; then \
	        echo "*** Multiple run_tests results logs from today found ***" ;\
	        exit 1 ;\
	    fi ;\
	    echo "Comparing $$logfile to bench/$$results_dir/pyferret_run_tests_log" ;\
	    diff $$logfile $$results_dir/pyferret_run_tests_log 1> pyferret_run_tests_log.diff 2>&1 ;\
	    errfile=`echo $$logfile | sed -e "s%log%err%"` ;\
	    echo "Comparing $$errfile to bench/$$results_dir/pyferret_run_tests_err" ;\
	    diff $$errfile $$results_dir/pyferret_run_tests_err 1> pyferret_run_tests_err.diff 2>&1 ;\
	    ncdfile=`echo $$logfile | sed -e "s%log%ncdump%"` ;\
	    echo "Comparing $$ncdfile to bench/$$results_dir/pyferret_run_tests_ncdump" ;\
	    diff $$ncdfile $$results_dir/pyferret_run_tests_ncdump 1> pyferret_run_tests_ncdump.diff 2>&1 ;\
	    echo '' ;\
	    echo 'Inspect the pyferret_run_tests_*.diff files for actual differences in the output' \
	)
	@echo ''

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
