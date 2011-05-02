#! /bin/csh -f
#! *sh* 10/91
#! Ftoc
#! browse the Table of Contents of the Ferret Users Guide

# enter the Table of Contents at the top
if ($#argv == "0") then
   more -d $FER_DIR/doc/ferret_ug_toc.txt
   exit
endif

# too many arguments: explain the syntax
if ($#argv >= 2) then
     echo " "
     echo "     *** Syntax error in command entered ***"
     echo " "
     echo "     Usage:  Ftoc    or    Ftoc  string"
     echo " "
     echo "     Note: multi-word strings need to be enclosed in quotations"
     echo " "
     exit
endif

# use grep for case-insensitive search
   grep -i "$argv[1]" $FER_DIR/doc/ferret_ug_toc.txt
