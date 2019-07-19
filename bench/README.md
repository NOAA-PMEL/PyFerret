## PyFerret benchmarks README file

Running the benchmarks and interpreting results. There are two methods for 
running the benchmarks.

The automated testing uses a script to start a new instance of pyferret to 
run each test, and does not display any plots. Expected results are given 
under the `bench/test_results` directory.

The manual testing uses a script to run all the tests in a single instance 
of pyferret. Plots are displayed as the tests are run. Expected results are
given by the `bench/ansley_official*` files.

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

### Manual/Displayed testing

After building PyFerret, make a few simple tests for command-line reading,
script reading, file reading, for example:

> $ pyferret  
> yes? list/L=1:5 L  
> yes? go ptest  
>  
> yes? use coads_climatology  
> yes? show data  

When PyFerret is running, the benchmark tests will verify its performance. To 
run the benchmarks, in the /bench directory we will run the script `run_all`. 
Look at the start of `run_all`, and if you like, customize the section for 
machine type. This information is used to make the log file, and also to look 
for a machine-specific stream binary file.

Run `run_all` and answer the questions. When you first run the shell script 
`run_all`, you may be coached to create a stream binary file for the machine 
by compiling and running the program `make_stream_file.F`. If so, do this, 
and then run `run_all` again. The benchmarks may be run with or without the 
shared-object external functions. If the benchmark scripts run correctly, 
the benchmark job will finish with:

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

In the benchmark directory are "official" output files.  To compare your output 
logs, choose one to compare with your output. There are lines in the benchmark 
output which may differ from one run of the benchmarks to another and which do not 
indicate problems with the benchmark run. We may remove them by running the script 
`clean_ultra` (or `clean_draconian`) and piping the output to a new file, for the 
official benchmark log file and the one just created:

> $ clean_ultra ansley_official.x86_64-linux_log \> cleaned_ansley_official.x86_64-linux_log  
> $ clean_ultra all_01nov16at0959.x86_64-linux_log \> cleaned_all_01nov16at0959.x86_64-linux_log  
> $ diff cleaned_ansley_official.x86_64-linux_log cleaned_all_01nov16at0959.x86_64-linux_log  

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

