#! /bin/csh -f
# Fpurge
# remove all but the current version of the indicated file(s)

# example usage:  Fpurge metafile.plt

# no argument given: explain the ropes
if ($#argv != "1") then
     echo '*** Syntax error - name 1 filename as template, only ***'
     echo '  Usage:  Fpurge  filename.extension'
     echo 'Example:  Fpurge ferret.jnl'
     exit
endif

rm {$argv[1]}.~*~
