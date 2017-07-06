#! /bin/sh -f
# run individually each of the benchmark tests listed in TEST_SCRIPTS

if [ $# -ne 2 ]; then
   echo ""
   echo "Usage:  $0  Ferret  ExtFuncsDir"
   echo "where:"
   echo "       Ferret is the ferret executable or pyferret script to use"
   echo "       ExtFuncsDir is the external functions directory to use"
   echo "           (if '.', bn_all_ef.jnl will not be run)"
   echo ""
   exit 1
fi

fver="$1"
efdir="$2"

PS1='$ '
export PS1

# allow tests to be commented out by beginning with the line with a '!'
# remove bn_all_ef.jnl from the list if $efdir is "."
if [ "$efdir" = "." ]; then
   jnl_scripts=`grep -v '^!' TEST_SCRIPTS | grep '\.jnl$' | grep -v "bn_all_ef\.jnl"`
else
   jnl_scripts=`grep -v '^!' TEST_SCRIPTS | grep '\.jnl$'`
fi
py_scripts=`grep -v '^!' TEST_SCRIPTS | grep '\.py$'`

umask 002

# Get the machine type for the stream file testing
if [ `uname -s` = "Linux" -a `uname -m` = "x86_64" ]; then
    machine="x86_64-linux"
elif [ `uname -s` = "Linux" -a `uname -m` = "i686" ]; then
    machine="linux"
elif [ `uname -s` = "Darwin" -a `uname -m` = "x86_64" ]; then
    machine="x86_64-darwin"
else
    echo "Unknown machine type"
    exit 1
fi

date_stamp=`date +'%d%h%yAT%H%M'|tr '[A-Z]' '[a-z]'`
log_file="all_${date_stamp}.${machine}_log"
err_file="all_${date_stamp}.${machine}_err"
ncdump_file="all_${date_stamp}.${machine}_ncdump"

# Make sure things are clean for this run
rm -f $log_file $err_file $ncdump_file
rm -f ferret.jnl* bat.plt* `cat TRASH_FILES`
rm -fr subdir tmp

touch $log_file $ncdump_file
touch F.cdf snoopy.dat

# set up the binary unformatted stream test file as a machine-specific link
machine_stream="stream10by5_${machine}.unf"
if [ -r  $machine_stream ]; then
   rm -f stream_data_link.unf
   ln -s $machine_stream stream_data_link.unf
else
   echo "File $machine_stream does not exist." >> $log_file
   echo "Benchmark bn420_stream will fail." >> $log_file
   echo "To create $machine_stream compile make_stream_file.F and run the executable" >> $log_file
   echo "(You may need to use the compile flag -Dreclen_in_bytes)" >> $log_file
   echo "Then rename stream10by5.unf to $machine_stream" >> $log_file
   echo "File $machine_stream does not exist."
   echo "Benchmark bn420_stream will fail."
   echo "To create $machine_stream compile make_stream_file.F and run the executable"
   echo "Then rename stream10by5.unf to $machine_stream"
fi

# set up proper stream testing jnl file - depends on endianness
# pretty much everything is little endian now
rm -f bn_test_stream.jnl
ln -s bn_test_stream_little.jnl bn_test_stream.jnl
# ln -s bn_test_stream_big.jnl bn_test_stream.jnl

echo "Testing log output in $log_file"
echo "Testing errors in $err_file" 
echo "Testing ncdump output in $ncdump_file"

if ! echo "$fver" | grep -q "pyferret"; then
   ispyferret=0
#  command-line options for ferret
   feropts="-nojnl -gif -noverify"
#  external functions search path
   FER_EXTERNAL_FUNCTIONS="$efdir"
   export FER_EXTERNAL_FUNCTIONS
else
   ispyferret=1
#  command-line options for pyferret
   feropts="-nojnl -quiet -nodisplay -noverify -linebuffer"
#  external functions search path
   PYFER_EXTERNAL_FUNCTIONS="$efdir"
   export PYFER_EXTERNAL_FUNCTIONS
fi

# set up a generic data environment
echo "****** Restricting Ferret paths to bench directory ******" >> $log_file
FER_DATA="."
export FER_DATA
FER_DESCR="."
export FER_DESCR
FER_DSETS="."
export FER_DSETS
FER_DAT="."
export FER_DAT
FER_GRIDS="."
export FER_GRIDS
FER_DIR="."
export FER_DIR

# always replace $HOME/.ferret with default.ferret so results are consistent
rm -f keep.ferret
if [ -f $HOME/.ferret ]; then 
   echo "****** Temporarily moving $HOME/.ferret to ./keep.ferret ******"
   mv -f $HOME/.ferret ./keep.ferret
fi
cp ./default.ferret $HOME/.ferret

echo "Benchmark scripts that will be run:" >> $log_file
for script in $jnl_scripts; do
   echo "   $script" >> $log_file
done
if [ "$ispyferret" -ne 0 ]; then
   for script in $py_scripts; do
      echo "   $script" >> $log_file
   done
fi

# run each of the scripts in the list
rm -f all_ncdump.out
for script in $jnl_scripts; do

   echo "*** Running ferret script: $script" >> $log_file
   echo "*** Running ferret script: $script" >> $err_file
   echo "*** Running ferret script: $script" > all_ncdump.out
   echo "Running ferret script: $script"

   if [ $script = "bn_startupfile.jnl" ]; then
#     bn_startupfile.jnl needs ferret_startup as $HOME/.ferret
      rm -f $HOME/.ferret
      cp -f ferret_startup $HOME/.ferret
   fi

   if [ $script = "bn_dollar.jnl" ]; then
      $fver $feropts -script $script hello 1>> $log_file 2>> $err_file
   else
      $fver $feropts -script $script 1>> $log_file 2>> $err_file
   fi
   if [ $? -ne 0 ]; then
      echo "****** FERRET error: $script failed ******" >> $log_file
      echo "****** FERRET error: $script failed ******" >> $err_file
      echo "****** FERRET error: $script failed ******" >> all_ncdump.out
      echo "****** FERRET error: $script failed ******"
   fi

   if [ $script = "bn_startupfile.jnl" ]; then
#     remove the $HOME/.ferret created for bn_startupfile.jnl
      rm -f $HOME/.ferret
      cp -f default.ferret $HOME/.ferret
   fi

#  add the contents of all_ncdump.out to $ncdump_file
   cat all_ncdump.out >> $ncdump_file
   rm -f all_ncdump.out
   rm -f ferret.gif
   rm -f ferret.png
done

if [ "$ispyferret" -ne 0 ]; then
   for script in $py_scripts; do
      echo "*** Running python script: $script" >> $log_file
      echo "*** Running python script: $script" >> $err_file
      echo "*** Running python script: $script" > all_ncdump.out
      echo "Running python script: $script"
      $fver $feropts -python < $script 1>> $log_file 2>> $err_file
      if [ $? -ne 0 ]; then
         echo "****** PYFERRET error: $script failed ******" >> $log_file
         echo "****** PYFERRET error: $script failed ******" >> $err_file
         echo "****** PYFERRET error: $script failed ******" >> all_ncdump.out
         echo "****** PYFERRET error: $script failed ******"
      fi
      cat all_ncdump.out >> $ncdump_file
      rm -f all_ncdump.out
   done
fi

# Replace $HOME/.ferret if it was removed
rm -f $HOME/.ferret
if [ -f keep.ferret ]; then
   echo "****** Returning keep.ferret to $HOME/.ferret ******"
   mv keep.ferret $HOME/.ferret
fi

# Replace insignificant differences with constant values
cleanups="cleanups.sed"

builddir=`dirname $PWD | sed -e 's/\\//\\\\\\//g'`
echo "s/$builddir/....../g" > $cleanups

exebindir=`dirname $fver`
exeferdir=`dirname $exebindir | sed -e 's/\\//\\\\\\//g'`
echo "s/$exeferdir/....../g" >> $cleanups

timeregex=`date +%_d.%h.%Y`
echo "s/$timeregex/DD-MON-YYYY/g" >> $cleanups
# If date assigned to symbol and then symbol used elsewhere, any beginning space is dropped
timeregex=`date +%-d.%h.%Y`
echo "s/$timeregex/DD-MON-YYYY/g" >> $cleanups

timeregex=`date +%_d.%h.%y`
echo "s/${timeregex}.[0-9][0-9]:[0-9][0-9]/DD-MON-YY HH:MM/g" >> $cleanups
echo "s/$timeregex/DD-MON-YY/g" >> $cleanups
# If date assigned to symbol and then symbol used elsewhere, any beginning space is dropped
timeregex=`date +%-d.%h.%y`
echo "s/${timeregex}.[0-9][0-9]:[0-9][0-9]/DD-MON-YY HH:MM/g" >> $cleanups
echo "s/$timeregex/DD-MON-YY/g" >> $cleanups

timeregex=`date +%m.%d.%y`
echo "s/$timeregex/MM-DD-YY/g" >> $cleanups

timeregex=`date | sed -e 's/[0-9][0-9]:[0-9][0-9]:[0-9][0-9]/[0-9][0-9]:[0-9][0-9]:[0-9][0-9]/'`
echo "s/$timeregex/WKD MON DD HH:MM:SS ZZZ YYYY/g" >> $cleanups

timeregex=`date +%a.%h.%_d.%T.%Y | sed -e 's/[0-9][0-9]:[0-9][0-9]:[0-9][0-9]/[0-9][0-9]:[0-9][0-9]:[0-9][0-9]/'`
echo "s/$timeregex/WKD MON DD HH:MM:SS YYYY/g" >> $cleanups

echo 's/^randu2_randn2 [0-9 .-]+$/randu2_randn2      ....../' >> $cleanups
echo 's/the_time = [0-9][0-9]:[0-9][0-9]/the_time = HH:MM/g' >> $cleanups
echo 's/\(AX[0-9][0-9][0-9]\)/(AX###)/g' >> $cleanups
echo 's/\(G[0-9][0-9][0-9]\)/(G###)/g' >> $cleanups
echo 's/CURRENT_TIME = "[0-9][0-9]:[0-9][0-9]:[0-9][0-9]"/CURRENT_TIME = "HH:MM:SS"/g' >> $cleanups
echo 's/SESSION_TIME = "[0-9][0-9]:[0-9][0-9]"/SESSION_TIME = "HH:MM"/g' >> $cleanups
echo 's/SESSION_PID = "[0-9]+"/SESSION_PID = "#####"/g' >> $cleanups
echo 's/DELTA_CPU = "[0-9]\.[0-9E-]+"/DELTA_CPU = "######"/g' >> $cleanups
echo 's/CLOCK_SECS = "[0-9]\.[0-9E-]+"/CLOCK_SECS = "######"/g' >> $cleanups
echo 's/^\[\?1034h//' >> $cleanups
echo 's/Second 10K LET commands LET a = 0 takes  [0-4]\.[0-9]+  seconds/Second 10K LET commands LET a = 0 takes [0-5] seconds/' >> $cleanups
echo 's/10K LET commands LET a = 0 takes  [0-2]\.[0-9]+  seconds/10K LET commands LET a = 0 takes [0-3] seconds/' >> $cleanups
echo 's/5K LOAD with transform takes  [0-8]\.[0-9]+  seconds/5K LOAD with transform takes [0-9] seconds/' >> $cleanups
echo 's/DEFINE VARIABLE ten_plots = 0\.[0-9]+/DEFINE VARIABLE ten_plots = 0.######/' >> $cleanups
echo 's/DEFINE VARIABLE dt = 0\.[0-9]+/DEFINE VARIABLE dt = 0.######/' >> $cleanups
echo 's/DEFINE VARIABLE sumcpu =[ ]?0\.[0-9]+/DEFINE VARIABLE sumcpu = 0.######/' >> $cleanups
echo '/say `sumcpu`/,+1 s/^ !-> MESSAGE\/CONTINUE 0\.[0-9]+$/ !-> MESSAGE\/CONTINUE 0.######/' >> $cleanups
echo '/say `sumcpu`/,+2 s/^0\.[0-9]+$/0.######/' >> $cleanups

sed -r -i_orig -f $cleanups $log_file
sed -r -i_orig -f $cleanups $err_file
sed -r -i_orig -f $cleanups $ncdump_file

rm -f $cleanups

# Clean-up
rm -f `cat TRASH_FILES`
rm -fr subdir tmp

