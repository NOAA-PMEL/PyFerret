## PyFerret benchmarks README file for developers

See the README.md file in the bench/ directory as well.


### Automated testing

1) Build and install
Build pyferret using `make debug` , install it, and set the environment variables 
using the appropriate `ferret_paths` script.  `make debug` is used when comparing
benchmark output, to make it easier to check on progress during debugging.

2) Run benchmark suite

From the pyferret directory, `make check` will run the benchmark suite, executing the 
benchmark test journal files, each in a new PyFerret session.  This is done via executing
`make run_tests` from the bench directory, then comparing graphics and log files. Pipe
the output of `make check` to a file, and examine that file for reports of any benchmark 
scripts that may have failed.

There is a version of the benchmark suite which does not require any remote datasets 
(OPeNDAP datasets).  If OPeNDAP is unavailable, run `make check_noremote`.

3) Comparing graphics files

If the file size of any graphics file differs from the file size of the standard comparison 
graphics files, ImageMagick compare utility is used to create `*_diff.png` file to show the
difference. These should be examined to see if there are any real differences in the plot, 
which are shown in bright solid red against a faded image of the original plot).  

4) Comparing log files

`make check` includes a comparison of the log files.  The text output from the scripts are 
collected in a file named according to the date and time of the benchmark run, e.g. 

all_01nov16at0959.x86_64-linux_err
all_01nov16at0959.x86_64-linux_log
all_01nov16at0959.x86_64-linux_ncdump

You will also see files of the same names ending with _orig.  These are the files created 
from the benchmark run. The above are versions of the files for comparison have been 
cleaned up to remove things like the current date, which will always differ from the 
expected results.  Compare with the expectedresults using diff or perhaps  meld,

> diff test_results/pyferret_run_tests_err all_01nov16at0959.x86_64-linux_err
> diff test_results/pyferret_run_tests_ncdump all_01nov16at0959.x86_64-linux_ncdump
> meld test_results/pyferret_run_tests_log all_01nov16at0959.x86_64-linux_log


5) Investigating differences

To run individual benchmark scripts that may have failed or returned unexpected results
or graphics output, execute the bench_environment.csh or bench_environment.sh script 
This changes the Ferret environment variable to point to the datasets and journal files that 
make up the benchmark suite so that from the bench directory, one may run the scripts,
e.g. `yes? go bn_plot`

6) Adding new tests

When a new feature is added or a bug is fixed, create one or more .jnl scripts to test the change.  
Any datasets should be kept short, and are put into the bench/data directory; or use datasets
that are already in that directory. The script should be named bn_pyferret_feature.jnl for tests
of a feature, or err00_issue_description.jnl for a bug fix where the numbers are the current
running version (so the next release will include the fix).  Add the err*.jnl script to the current
version of the bnxx_bug_fixes.jnl, e.g. a new script  err751_use_order.jnl fixing a bug in V7.51 of
the code is part of the tests for version 7.60, so `go err751_use_order.jnl` is added to the script
bn76_bug_fixes.jnl.  After a release, if bugs or issues are found, start a new bnxx_bug_fixes.jnl
script (e.g. bn761_bug fixes.jnl, even though the next official release may not have that version number).

If the script creates graphics output, add the graphics files to the directory test_results and the 
directory test_results_noremote.  Add the script name to the lists of scripts in TEST_SCRIPTS and 
TEST_SCRIPTS_NOREMOTE

7) Preparing a release

When a new PyFerret release is to be made, update the standard log files in test_results and 
test_results_noremote, using a build of PyFerret created using `make clean; make debug`. 
Copy the log files generated from a run of the benchmarks, e.g.

all_01nov16at0959.x86_64-linux_err
all_01nov16at0959.x86_64-linux_log
all_01nov16at0959.x86_64-linux_ncdump

to

test_results/pyferret_run_tests_err 
test_results/pyferret_run_tests_log
test_results/pyferret_run_tests_ncdump 




### Manual/Displayed testing

The alternative method is to run all of the benchmark tests in a single
PyFerret session, using the script `bench/run_all`.   This tests the code
differently.  We have not kept up with this  means of testing, and it is
no longer described in the README.md file, but it is kept here, as it may be
a way to investigate bugs that are hard to replicate with simple examples but
are appearing in PyFerret sessions that run complex sets of commands or
multiple scripts.

Add new benchmark scripts to the master script bench/genjnls/run_all.jnl and
graphics output files to the directory bench/runall_master_plots

From the bench directory, run `run_all` and answer the questions. When you first 
run the shell script `run_all`, you may be coached to create a stream binary file for 
the machine by compiling and running the program `make_stream_file.F`. If so, do 
this, and then run `run_all` again. The benchmarks may be run with or without the 
shared-object external functions. If the benchmark scripts run correctly, the 
benchmark job will finish with:

> Display no longer set, about to run batch gif test  
> Display still not set, about to run batch ps test  
> Display still not set, about to run batch metafile test
> Ended at (some date and time)  

The output is contained in two files:

> all_01nov16at0959.x86_64-linux_log  
> all_01nov16at0959.x86_64-linux_err  

where the name contains the date and time of the benchmark run, and the extension 
refers to the machine type or operating system. In addition a number of plot 
output files are created and compared to reference output by the benchmark script.

In the benchmark directory are `official` output files.  To compare your output 
logs, choose one to compare with your output. There are lines in the benchmark 
output which may differ from one run of the benchmarks to another and which do not 
indicate problems with the benchmark run. We may remove them by running the script 
`clean_ultra` (or `clean_draconian`) and piping the output to a new file, for the 
official benchmark log file and the one just created:

> $ clean_ultra run_all_logs.x86_64-linux_log \> cleaned_run_all_logs.x86_64-linux_log  
> $ clean_ultra all_01nov16at0959.x86_64-linux_log \> cleaned_all_01nov16at0959.x86_64-linux_log  
> $ diff cleaned_run_all_logs.x86_64-linux_log cleaned_all_01nov16at0959.x86_64-linux_log  

Some differences will still always exist: 

1. The date of the Ferret run and the operating system are included in various 
outputs such as the Ferret symbol `SESSION_DATE`, values of labels which are 
written to the logs, or file attributes which are listed. These differences 
may be ignored.

2. Values of `PPL$XPIXEL` and `PPL$YPIXEL` will differ; these are computed 
based on the display of the terminal where the benchmark job is run.

3. If you are comparing a log from a different operating system, there 
may be differences in the values of data in output. This might show up as 
`missing_value=-9.9999998e+33f` vs `missing_value=-1.e-34f`, 
or listings may differ in the least-significant positions. Differences of 
that size are okay. 

4. At the end of the log files, there is a collection of outputs from 
`spawn ncdump file.nc` commands.  Differences in the form of ncdump output, 
such as differently-placed commas, may exist especially if you are comparing 
logs from different operating systems.

5. Some benchmark scripts involve the output of a spawn command. The speed with 
which this output is written to the log file may vary from run to run or from 
system to system. Occasional garbled output is the result of this effect. 

6. PyFerret generates PostScript (.ps) files where Ferret generates .plt files.

