# frequency.awk - *sh* 8/94
# this procedure should be executed with awk -f frequency.awk input.file
# The input file should be a sorted list of records with 2 values per
# record.  The first value should be the sorting index - the second should
# be a weight to associate with that record.  The sorting index may
# contain duplicates and gaps.
# The output will be a list of consecutive integers and the corresponding
# sum of weights

# due to what appears to be a bug in awk the diagnostic print line "XXXXX"
# is needed for proper functioning.

# Here is the complete usage pipe for an unsorted list:
# sort -n myintegers.dat | awk -f TS_frequency.awk | grep -v "XXXXX"

#printf("XXXXXXXXX  %d %d\n",$1,$2)   # doesn't work w/out this ???

BEGIN   {ndx=1
         wt=0}

	{
          if ($1 == 0 ) {
            ndx = 1
          }
          else {
            if ($1 == ndx) {
            wt = wt + $2
	    } else {
	      if (ndx==1) wt=0       # index 1 used to capture garbage
	      printf("%d   %f\n",ndx,wt)
              wt = 0
	      for (i=ndx+1; i<$1; i++) printf("%d   %f\n",i,wt)
	      ndx=$1
              wt=$2
	    }
          }
	}

