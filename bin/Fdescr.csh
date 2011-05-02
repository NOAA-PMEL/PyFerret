#! /bin/csh -f
# Fdescr file_template    
# determine if FERRET descriptor files matching template are currently on-line
# by searching the paths in FER_DESCR

# 8/92 bug fix: on SUNs /bin/test can accept only one arg.  Use nonomatch
# to resolve the list of files matching template and pass only one name to test
# 21mar94 *kob* Solaris port -----
#		  /bin/test doesn't exist on solaris (sunos 5.x) so had to 
#		  do a check for that OS and then point it to /usr/ucb/test
# 30may97 *kob* Linux port - test is in /usr/bin/test


if ($#argv == 0 ) then
   echo "usage: Fdescr descriptor_file_template"
   exit
endif

#check for sunos 5.x
if (`uname` =~ *Sun* && `uname -r` =~ *5.*) then
	set TEST = /usr/ucb/test
else if (`uname` =~ *inux* ) then
	set TEST = /usr/bin/test
else
	set TEST = /bin/test
endif


set nonomatch
set found = 0

foreach path ($FER_DESCR)
   cd $path
   set flist = *$argv*
    $TEST -f $flist[1] >& /dev/null
    if ($status == 0) then    
      echo "* * * * * * * * in $path"
      /bin/ls -l *$argv*
      set found = 1
      echo " " 
   endif
end

if ( $found == 0 ) then
   echo "No files matching $argv are on line"
endif

