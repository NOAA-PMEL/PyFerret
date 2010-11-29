# Fapropos.awk
# intended for use in a pipe following "grep -n"
# grep -n begins each line with nn: where nn is the line number
# Usage:  grep -n string file | awk -f this_file

BEGIN	{ FS = ":" }

# header for output
	{ if ( NR==1 ) { print " LINE       TEXT"
	                  print " ----     --------" }
	}

# formatted output lines
	{ printf "%5d: ",$1
	  for (i=2; i<=NF; i++) { printf "%s",$i }
	  printf "\n"
	}