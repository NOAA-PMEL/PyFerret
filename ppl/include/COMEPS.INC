c
c	comeps
C** 
C**    @(#)COMEPS.INC	1.1    3/10/88
C**
C**
C***********************************************************************        C**
C**                 EPIC
C**
C***********************************************************************        C**
C**
c
	common /comeps/ lstime,lsoffs,lsctd,lsnext
	logical lstime,lsoffs,lsctd,lsnext
c
c**
c
c	lstime = .false. for EVAR/NOTIME (default) meaning to read the 
c                        start time and delta-t from EPIC time series
c                        headers and not use the time words on the file.
c              = .true.  for EVAR/TIME meaning to read the time words
c                        from the data file (for data which is not
c                        evenly spaced in time).
c	lsoffs = .true.  for EVAR/OFFSET (default) to offset EPIC time 
c	                 series data so that data point is plotted at 
c	                 center of time interval.
c	       = .false  for EVAR/NOOFFSET meaning to plot data at
c	                 the time on the data file (without offsetting
c	                 to the center of the time interval).
c	lsctd  = .true.  for FORMAT/CTD or EVAR/CTD or VAR/CTD meaning
c	                 that the data is EPIC CTD data.
c	         .false. for /NOCTD (default)
c	lsnext = .true.  for EVAR/NEXT meaning that the next x variable
c                        should be read from the same data file as the
c	                 last data set was read from.  In this case, the
c	                 y-variable must be the same as it was for the
c                  last read.  This feature enables making CTD data
c                        plots with multiple x-axes (e.g., P vs 
c	                 T,S,sigma-T, Oxygen).  Only enabled for 
c	                 EPIC CTD data.
c	         .false. for /NONEXT (default)
c
c***********************************************************************
