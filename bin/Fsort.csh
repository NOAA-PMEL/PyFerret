#! /bin/csh -f
# sort a list of names given as arguments into increasing
# order based on their ~nnn~ version number extensions
# at most one name should be without ~nnn~

# example usage:  Fsort metafile.plt*
# example in context:	animate `Fsort metafile.plt*`
# ... or in a more complex example
#	animate start_frame `Fsort series1.plt*` `Fsort series2.plt*` end_frame

echo $argv | nawk -f $FER_DIR/bin/Fsort.nawk

# Note:
# Here is an alternative version that uses sort for the "tilda" filenames
# and sed to put the filename with no tilda version number at the end:
# ls !* | sort -t\~ +1n | sed -e '/\~/\!h' -e '/\~/\!d' -e '$p' -e '$x'

