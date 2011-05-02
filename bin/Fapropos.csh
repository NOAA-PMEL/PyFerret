#! /bin/csh -f
#! *sh* 10/91
#! Fapropos
#! scan the Ferret Users Guide for the indicated string
#! report occurrances with line numbers
#! the command Fhelp can then be used to access the Usegs Guide
#! beginning at a selected line number

# no argument given: explain the ropes
if ($#argv == "0") then
     echo ' '
     echo '	*** Fapropos - Interactive help for FERRET ***'
     echo ' '
     echo '	Fapropos scans the FERRET Users Guide for a character string'
     echo ' '
     echo '		Correct usage is:  Fapropos  string'
     echo ' '
     echo '		Example: Fapropos regridding'
     echo ' '
     echo '	Fhelp can then be used to enter the Users Guide'
     echo '		Correct usage is:  Fhelp line_number'
     exit
endif

# too many arguments: explain the syntax
if ($#argv >= 2) then
     echo " "
     echo "	*** Syntax error in command entered ***"
     echo " "
     echo "	Usage:  Fapropos  string"
     echo " "
     echo "	Note: multi-word strings need to be enclosed in quotations"
     exit
endif

# scan the FERRET manual
grep -in "$argv" $FER_DIR/doc/ferret_users_guide.txt | awk -f $FER_DIR/bin/Fapropos.awk | more -d
switch ($status)
case "1":
   echo " $argv is not found in the FERRET manual"
   breaksw
case "2":
   echo "Syntax error in string entered"
   echo "Multiword strings need surrounding quotations"
endsw
