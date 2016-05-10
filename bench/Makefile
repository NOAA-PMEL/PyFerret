#
# Makefile for the FERRET/bench directory
#

## Execute the RUN_TESTS.sh test suite
.PHONY: run_tests
run_tests:
	@if [ `which ferret` = "" -o "$(FER_DIR)" = "" ]; then \
	    echo "*** Ferret must be installed and the ferret_paths script must have sourced for these tests ***" ; \
	    exit 1 ; \
	fi
	./RUN_TESTS.sh `which ferret` "$(FER_EXTERNAL_FUNCTIONS)" "RUN_TESTS.sh test results for Ferret"
	@echo 'Differences in the generated bench/*.gif files versus against those in bench/test_results'
	@for img in *.gif ; \
	do \
	    diff -q $$img test_results/$$img ; \
	done
	@echo ''
	@echo 'Inspect the bench/all_*_ncdump, bench/all_*_err, and bench/all_*_log files just created '
	@echo 'against bench/test_results/ferret_run_tests_ncdump, bench/test_results/ferret_run_test_err, '
	@echo 'and bench/test_results/ferret_run_tests_log'

.PHONY: clean
clean:
	rm -f all_*_ncdump all_*_err all_*_log
	rm -f *.plt* *.gif* *.ps* *.png* *.pdf* ferret.jnl*
	rm -f `cat TRASH_FILES`
	rm -fr subdir tmp

#
# End of Makefile
#
