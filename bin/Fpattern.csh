#! /bin/csh -f
# Fpattern file_template    
# determine if FERRET pattern files matching template are currently on-line
# by searching the paths in FER_DESCR

# 8/92 bug fix: on SUNs /bin/test can accept only one arg.  Use nonomatch
# to resolve the list of files matching template and pass only one name to test
#
# utterly modified for osf port.  1.29.94 *kob* Allow inclusion of 
#  "-help", "-l", and "-more" options.  Also allows for desired file
# to end in ".spk"
# 21mar94 *kob* Solaris port -----
#		  /bin/test doesn't exist on solaris (sunos 5.x) so had to 
#		  do a check for that OS and then point it to /usr/ucb/test
# 18may94 *kob* removed the "ls -1" and made it "ls" because of lack of "-1"
#		option on SGI's
# 18dec98 *jcd* Hack to look for .pat (pattern) files rather than .spk

#check for proper amount of args.  One arg is the filename or template. 
if ($#argv == 0 || $#argv > 2) then
  echo " "
  echo "Usage: Fpattern [-help] [-l] [-more]  pattern_file[_template]"
  echo "Type Fpattern -help for a full description"
  echo " "
  exit 1
endif

# print out help message
if ("$argv[1]" =~ *hel* || "$argv[1]" == "-h") then
usage:
  echo " "
  echo "Usage:"
  echo "        Fpattern [-help] [-l] [-more]  pattern_file[_template]"
  echo  " "
  echo "where options include: "
  echo "   -help	print this message, option not valid with any other"
  echo "   -l		generate long listing, without description of pattern"
  echo "   -more	display files matching given template using more"
  echo " "
  echo "These options precede either the pattern file, if it is known,"
  echo "or a pattern file template.  Files found matching the given template"
  echo "are then listed, or more'd if the -more option is passed. All options"
  echo "are mutually exclusive. To see all of the Pattern  files"
  echo "available, enter: "
  echo "       Fpattern '*'"
  echo "It is important to have the single quotes around the asterisk."
  echo " " 
  exit 1
endif


# set some variables
set num_args = $#argv
set nonomatch
set found = 0
#set some commands that seem to wander on various systems
set GREP = /bin/grep
set EGREP = /usr/bin/egrep
set SED = /bin/sed


#check for sunos 5.x
if (`uname` =~ *Sun* && `uname -r` =~ *5.*) then
	set TEST = /usr/ucb/test
else if (`uname` =~ *inux* ) then
	set TEST = /usr/bin/test
        set EGREP = /bin/egrep
else
	set TEST = /bin/test
endif



# check to see if file contains .pat or not
if ("$argv[$num_args]" =~ *.pat*) then
	set tag = 1
else
	set tag = 0
endif

# if there is only one argument, it must be the file name, otherwise it
# is a usage error
if ($num_args == 1) then
#check for usage error
  if ("$argv[1]" =~ *-l* || "$argv[1]" =~ *-hel* || "$argv[1]" =~ *-m*) goto usage
  foreach path ($FER_PALETTE)
	cd $path
# check for existance of an extension.  If no extension, apply .pat default
	if ($tag) then
		set flist = *$argv*
	else
		set flist = *$argv*.pat
	endif
	$TEST -f $flist[1]
	if ($status == 0) then   
      		echo "* * * * * * * * in $path"
		foreach file ($flist)
      		 echo `/bin/ls $file`: `$EGREP '[ ][dD][eE][sS][cC][rR][iI][pP][tT][iI][oO][nN]:[ ]' $file ` | $SED -e "s/\![ ][dD][eE][sS][cC][rR][iI][pP][tT][iI][oO][nN]:[ ]//"
		end
		set found = 1
      		echo " " 
   	endif
   end
	goto the_end
#if num_args is two, then we either have to do an ls -l, or a more.
#cannot do both.
else if ( $num_args == 2 ) then
#do a long listing
	switch ($argv[1]) 
	  case '*l*' :
		foreach path ($FER_PALETTE)
   			cd $path
   			set flist = *$argv[2]*
    			$TEST -f $flist[1] >& /dev/null
    			if ($status == 0) then    
      			   echo "* * * * * * * * in $path"
			   if ($tag) then
      			   	/bin/ls -l $argv[2]
			   else	
				/bin/ls -l *$argv[2]*.pat
			   endif
      			   set found = 1
      		 	   echo " " 
   			endif
		end
	     breaksw
	  case '*-m*':
# more each file we come across which matches the template.
		foreach path ($FER_PALETTE)
   			cd $path
   			set flist = *$argv[2]*
    			$TEST -f $flist[1] >& /dev/null
			if ($status == 0) then    
      			   echo "* * * * * * * * in $path"
			   if ($tag) then
      			   	/usr/ucb/more $argv[2]
			   else	
				/usr/ucb/more *$argv[2]*.pat
			   endif
      			   set found = 1
      		 	   echo " " 
   			endif
		end
	     breaksw
	  default:
	     goto usage
	endsw	     
 	goto the_end
endif


the_end:
if ( $found == 0 ) then
   if ($tag) then
	   echo "No files matching $argv are on line"
   else
	   echo "No files matching $argv.pat are on line"
   endif
endif

