#! /bin/sh
#
# shell version of run_all for Cygwin
# js

umask 002     #  make all files group deletable  3/16

# benchmark version to run ?

if [ $# = 1 ]
then
	bver=$1
else
	bver=bn500   
fi

machine="cygnus"

echo "Enter the path (including filename) to the FERRET version of choice"
read fver

# XGKS version ?  (use "xg" as the indicator)
echo $fver | grep -s "xg"
if [ $? -eq 0 ]
then
	machine="x$machine"
fi

# background info to go in the benchmark file
echo "Enter your name"
read bmarker
echo "Enter comment about this benchmark or this version of Ferret"
read bcomment

date_stamp=`date +'%d%h%yAT%H%M'|tr '[A-Z]' '[a-z]'`
log_file="all_${date_stamp}.${machine}_log"
err_file="all_${date_stamp}.${machine}_err"
plt_file="all_${date_stamp}.${machine}_plt"

# up the binary unformatted stream test file as a machine-specific link
machine_stream="stream10by5_${machine}.unf"
if [ ! -e  $machine_stream ]; then
   echo File $machine_stream does not exist.
   echo Benchmark bn420_stream will fail.
   echo To create $machine_stream compile and run make_stream_file.F
   echo Then rename stream10by5.unf to $machine_stream
   echo -n 'Continue anyway? (answer "y" for yes)'
   read answer
   if [ $answer != "y" ]
   then
    exit 0
   fi
fi
rm -f stream_data_link.unf
ln -s $machine_stream stream_data_link.unf

echo "Log output in $log_file  Errors in $err_file" 
echo "Procedure run_all to run all FERRET benchmarks" >$log_file 2>&1

echo "Running FERRET version $fver" >>$log_file 2>&1
ls -l $fver >>$log_file 2>&1
echo "Running benchmark version $bver" >>$log_file 2>&1
echo "Benchmark run by $bmarker" >>$log_file 2>&1
echo "Note: $bcomment" >>$log_file 2>&1

touch F.cdf snoopy.dat
temp_files="test.dat test.gt test.unf WV.J34K56L7 F*.cdf test*.cdf newjournal.jnl fort.41"
for file in $temp_files
do
      /bin/rm -f $file
done


# 10/97 *kob* mv last created bnplot files to the last_plot directory
#mv bnplot.* last_plot

now=`date`
echo "Beginning at $now" >>$log_file 2>&1
cp $log_file $err_file
echo "Beginning at $now"

($fver <${bver}_all_shell.jnl >>$log_file) >>$err_file 2>&1


# check status before continuing *kob* 4/98
if [ $? -ne 0 ]
then
    echo "FERRET ERRROR*********************************" >>$log_file
    exit 1
fi


# *kob* 11/96 - test out batch ability of ferret
#set hold_display = $DISPLAY
#setenv DISPLAY ralf
#echo "Display now set to "$DISPLAY >>$log_file 2>&1
#($fver < bn430_batch.jnl >> $log_file) >>$err_file 2>&1
#setenv DISPLAY $hold_display
#echo "Display now set to "$DISPLAY >>$log_file 2>&1

hold_display=$DISPLAY
DISPLAY=
export DISPLAY
echo "Display no longer set, about to run batch gif test"
($fver -gif <bn450_gif_shell.jnl >> $log_file) >> $err_file 2>&1
echo "Display still not set, about to run batch ps test"
($fver -batch bnplot.ps <bn450_ps_shell.jnl >> $log_file) >> $err_file 2>&1
DISPLAY=$hold_display
export DISPLAY
#echo "Display now set to "$DISPLAY >>$log_file 2>&1

now=`date`
echo  "Ended at $now" >>$err_file 2>&1
echo  "Ended at $now" >>$log_file 2>&1
echo  "Ended at $now"

# *kob* 8/96
sed 's/@ASFERRET .*Ver.*$/@ASFERRET ... whatever version number .../' <fort.41 >$plt_file


#clean_bench_log $log_file
#echo "Remember to remove metafiles, if any"


