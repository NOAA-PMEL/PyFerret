#! /bin/sh -f
# run individually each of the benchmark tests listed in TEST_SCRIPTS

if [ $# -lt 2 ]; then
   echo ""
   echo "Usage:  $0  Ferret  ExtFuncsDir  [ ... ]"
   echo "where:"
   echo "       Ferret is the ferret executable or pyferret script to use"
   echo "       ExtFuncsDir is the external functions directory to use"
   echo "           (if '.', bn_all_ef.jnl will not be run)"
   echo "       Any remaining arguments are used as comments in the log files"
   echo ""
   exit 1
fi

fver="$1"
shift
efdir="$1"
shift
bmarker="$USER"
bcomment="$*"

# allow tests to be commented out by beginning with the line with a '!'
# remove bn_all_ef.jnl from the list if $efdir is "."
if [ "$efdir" = "." ]; then
   test_scripts=`grep -v '^!' TEST_SCRIPTS | grep -v "bn_all_ef\.jnl"`
else
   test_scripts=`grep -v '^!' TEST_SCRIPTS`
fi

umask 002

# Get the machine type for the stream file testing
if [ `uname -s` = "Linux" -a `uname -m` = "x86_64" ]; then
    machine="x86_64-linux"
elif [ `uname -s` = "Linux" -a `uname -m` = "i686" ]; then
    machine="linux"
else
    echo "Unknown machine type"
    exit 1
fi

date_stamp=`date +'%d%h%yAT%H%M'|tr '[A-Z]' '[a-z]'`
log_file="all_${date_stamp}.${machine}_log"
err_file="all_${date_stamp}.${machine}_err"
ncdump_file="all_${date_stamp}.${machine}_ncdump"
rm -f $log_file $err_file $ncdump_file
touch $log_file $ncdump_file

# set up the binary unformatted stream test file as a machine-specific link
machine_stream="stream10by5_${machine}.unf"
if [ -r  $machine_stream ]; then
   rm -f stream_data_link.unf
   ln -s $machine_stream stream_data_link.unf
else
   echo "File $machine_stream does not exist." >> $log_file
   echo "Benchmark bn420_stream will fail." >> $log_file
   echo "To create $machine_stream compile make_stream_file.F and run the executable" >> $log_file
   echo "Then rename stream10by5.unf to $machine_stream" >> $log_file
   echo "File $machine_stream does not exist."
   echo "Benchmark bn420_stream will fail."
   echo "To create $machine_stream compile make_stream_file.F and run the executable"
   echo "Then rename stream10by5.unf to $machine_stream"
fi

#set up proper stream testing jnl file - depends on endianness
rm -f bn_test_stream.jnl
if [ $machine = "linux" -o $machine = "alp" -o \
     $machine = "x86_64-linux" -o $machine = "ia64-linux" ]; then
    ln -s bn_test_stream_little.jnl bn_test_stream.jnl
else
    ln -s bn_test_stream_big.jnl bn_test_stream.jnl
fi

echo "Testing log output in $log_file"
echo "Testing errors in $err_file" 
echo "Testing ncdump output in $ncdump_file"

echo "Using FERRET $fver" >> $log_file
ls -l $fver >> $log_file
echo "Using external functions from $efdir" >> $log_file
echo "Benchmark run by $bmarker" >> $log_file
echo "Note: $bcomment" >> $log_file
echo "Benchmark scripts that will be run:" >> $log_file
for jnl in $test_scripts; do
   echo "   $jnl" >> $log_file
done

#set up external functions search path
FER_EXTERNAL_FUNCTIONS="$efdir"
export FER_EXTERNAL_FUNCTIONS

#set up a generic data environment
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
Fenv >> $log_file

# Make sure things are clean for this run
rm -f ferret.jnl* bat.plt* `cat TRASH_FILES`
rm -fr subdir
touch F.cdf snoopy.dat

now=`date`
echo "Beginning at $now" >> $log_file
cp $log_file $err_file
echo "Beginning at $now"

# always replace $HOME/.ferret with default.ferret so results are consistent
rm -f keep.ferret
if [ -f $HOME/.ferret ]; then 
   echo "****** Temporarily moving $HOME/.ferret to keep.ferret ******"
   mv -f $HOME/.ferret keep.ferret
fi
cp ./default.ferret $HOME/.ferret

if ! echo "$fver" | grep -q "pyferret"; then
#  command-line options for ferret
   feropts="-noverify"
else
#  command-line options for pyferret
   feropts="-quiet -nodisplay -noverify"
fi

# run each of the scripts in the list
rm -f all_ncdump.out
for jnl in $test_scripts; do

   echo "*** Running test: $jnl" >> $log_file
   echo "*** Running test: $jnl" >> $err_file
   echo "*** Running test: $jnl" > all_ncdump.out
   echo "Running test: $jnl"

   if [ $jnl = "bn_startupfile.jnl" ]; then
#     bn_startupfile.jnl needs ferret_startup as $HOME/.ferret
      rm -f $HOME/.ferret
      cp -f ferret_startup $HOME/.ferret
   fi

   if [ $jnl = "bn_dollar.jnl" ]; then
      $fver $feropts -script $jnl hello 1>> $log_file 2>> $err_file
   else
      $fver $feropts -script $jnl 1>> $log_file 2>> $err_file
   fi
   if [ $? -ne 0 ]; then
      echo "****** FERRET error: $jnl failed ******" >> $log_file
      echo "****** FERRET error: $jnl failed ******" >> $err_file
      echo "****** FERRET error: $jnl failed ******" >> all_ncdump.out
      echo "****** FERRET error: $jnl failed ******"
   fi

   if [ $jnl = "bn_startupfile.jnl" ]; then
#     remove the $HOME/.ferret created for bn_startupfile.jnl
      rm -f $HOME/.ferret
      cp -f default.ferret $HOME/.ferret
   fi

#  add the contents of all_ncdump.out to $ncdump_file
   cat all_ncdump.out >> $ncdump_file
   rm -f all_ncdump.out

done

# Replace $HOME/.ferret if it was removed
rm -f $HOME/.ferret
if [ -f keep.ferret ]; then
   echo "****** Returning keep.ferret to $HOME/.ferret ******"
   mv keep.ferret $HOME/.ferret
fi

# Clean-up
rm -f ferret.jnl* bat.plt* `cat TRASH_FILES`
# Remove temporary subdirectory
rm -fr subdir
# Remove links made by this script (not in TRASH_FILES)
rm -f bn_test_stream.jnl
rm -f stream_data_link.unf

now=`date`
echo  "Ended at $now" >> $err_file
echo  "Ended at $now" >> $log_file
echo  "Ended at $now"

