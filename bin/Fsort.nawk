# Fsort.nawk
# nawk routine to sort files with ~nnn~ endings into increasing order
# this routine presumes that all files except at most 1 will have
# the ~nnn~ ending
# input should be a list of names - all on a single record
# output will be similar

# syntax:  nawk -f Fsort.awk
# example: echo metafile.plt* | nawk -f Fsort.awk

BEGIN     { maxnum = 0 }

# save the names in an array "name" indexed by the nn value in ~nn~
# save the single name without ~nn~ as index 0
     { for ( i=1; i<=NF; i++ )
	{ {pos = match($i,/~[0-9]*~/)}
	   { if (pos==0)
	        { name[0] = $i }
	     else
	        { num = substr($i,pos+1,RLENGTH-2)
                  if (num>maxnum) {maxnum = num}
	          name[num] = $i }
	   }
	}
     }


# spit the names back out in order - name lacking ~nnn~ last
# skip missing names
END     {
	 for (i=1; i<=maxnum; i++)
	   { if (length(name[i]) != 0 ) {printf "%s ",name[i]} }
	 if (length(name[0]) != 0 ) {printf "%s ",name[0]}
	 printf "\n"
	}

