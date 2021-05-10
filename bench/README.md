## PyFerret benchmarks README file

Running the benchmarks and interpreting results. There are two methods for 
running the benchmarks.

The automated testing uses a script to start a new instance of pyferret to 
run each test, and does not display any plots. Expected results are given 
under the `bench/test_results` directory.

The manual testing uses a script to run all the tests in a single instance 
of pyferret. Plots are displayed as the tests are run. Expected results are
given by the `bench/run_all_logs*` files.

Note that these tests expect the nco utility programs `ncdump` and `ncatted` 
to be found on the system path. If there programs do not exist the tests will 
still run results will differ, particularly all ncdump output will be missing.
The ImageMagick program `compare` is also used to compare plots.

### Automated testing

Build and install pyferret, including setting the ferret environment variables 
using the appropriate `ferret_paths` script. You may wish to check that 
pyferret is running properly, for example:

> $ pyferret  
> yes? use coads_climatology  
> yes? shade /l=5 sst  
> yes? quit  

To test, change to the `bench` subdirectory and enter `make run_tests`.  This 
will test whichever pyferret executable is first found on the system path 
(`which pyferret`) and tests the external functions given under the directory 
specified by `$PYFER_EXTERNAL_FUNCTIONS`. A message is output to the console 
when each test script is run. On completion, plots and output are compared to 
expected results, with messages output to the console, differences in plots 
(using the `compare` program) saved to `*_diff.png` files, and differences in 
output saved to `*.diff` files.

Ideally there will be no `*_diff.png` files because all the PNG and PDF plots
are completely identical to the expected results.  If any `*_diff.png` files
do exist, they should be examined to see if there are any real differences in
the plot, which are shown in bright solid red against a faded image of the 
original plot).  Also, ideally the `*.diff` files are empty; if not, the 
differences should be examined for anything of real significance.

To remove all the files generated from the testing, enter `make clean` when
in the bench subdirectory.
