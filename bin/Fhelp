#! /bin/csh -f
#! *sh* 10/91
#! Fhelp
#! enter the Ferret Users Guide at the indicated line number
#! If no line number is given then coach on usage

# no argument given: explain the ropes
if ($#argv == "0") then
     echo ' '
     echo '	*** Fhelp - Interactive help for FERRET ***'
     echo ' '
     echo '	Fhelp enters the FERRET Users Guide at a given line number'
     echo '	      or at the first occurrence of a given string'
     echo ' '
     echo '		Correct usage is:  Fhelp line_number'
     echo '		              or   Fhelp string'
     echo '	             For example:  Fhelp "getting started"'
     echo ' '
     echo '	When reading the Users Guide use standard "more" commands:'
     echo '	? = help   b = back 1 page   CR = next line  space = next page'
     echo ' '
     echo ' '
     echo '	Also available: Fapropos'
     echo '	Fapropos scans the FERRET Users Guide for a character string'
     echo '	and reports the lines where it occurs'
     echo ' '
     echo '		Correct usage is:  Fapropos  string'
     echo ' '
      exit
endif

# too many arguments: explain the syntax
if ( $#argv >= 2 ) then
     echo " "
     echo "	*** Syntax error in command entered ***"
     echo " "
     echo "		Correct usage is:  Fhelp line_number"
     echo " "
     echo ' '
     echo '	Use Fapropos to scan the FERRET Users Guide for a character string'
     echo '	and determine the lines where it occurs'
     echo ' '
     echo '		Correct usage is:  Fapropos  string'
     exit
endif

# did user enter a line number or a string
    echo $argv[1] | grep -s '^[0-9]*$'

# enter the FERRET manual 2 lines before the requested line
if ( $status == 0 ) then

#  line number
   @ argv[1] -= 2
   more -d +$argv[1] $FER_DIR/doc/ferret_users_guide.txt
   exit
else
#  string: use grep for case-insensitive search
   set line = `grep -in "$argv[1]" $FER_DIR/doc/ferret_users_guide.txt | head -1 | sed 's/^\([0-9]*\):.*$/\1/' `
   @ line -= 2
   more -d +{$line} $FER_DIR/doc/ferret_users_guide.txt

endif
