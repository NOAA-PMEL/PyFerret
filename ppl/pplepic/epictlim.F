	subroutine epictlim (xdate,xtime,xdat2,xtim2)
C** 
C**    @(#)epictlim.F	1.1    3/10/88
C**
C**
C***********************************************************************        C**
C**                 EPIC
C**
C***********************************************************************        C**
C**
c
c  Called by PLOT5's RDSEQ.  Input is starting xdate,xtime and
c  ending xdat2,xtim2 for EPIC time series.  Output is itmin,itmax
c  which is range of times for all time series currently in buffer.
c
c  Programmed by N. Soreide, May 86.
c
#ifdef unix
	include 'pplinc/AXISL.INC'
	include 'pplinc/TAXIS.INC'
	include 'pplinc/DATA.INC'
#else
	include 'pplinc:axisl.inc'
	include 'pplinc:taxis.inc'
	include 'pplinc:data.inc'
#endif
c
	if (linen .eq. 1) then
c
c  First series in buffer
c
	    call epictimc (xdate,xtime,itmin)
	    if (iautot .eq. 1) itstrt=itmin
	    call epictimc (xdat2,xtim2,itmax)
	else
c
c  Find time range minimum
c
	    read (itmin,100) xd,xt
100	    format (f6.0,f4.0)
	    if (xdate .lt. xd) then
	        call epictimc (xdate,xtime,itmin)
	    else if (xdate.eq.xd .and. xtime.lt.xt) then
		call epictimc (xdate,xtime,itmin)
	    endif
c
c  Find time range maximum
c
	    read (itmax,100) xd,xt
	    if (xdat2 .gt. xd) then
		call epictimc (xdat2,xtim2,itmax)
	    else if (xdat2.eq.xd .and. xtim2.gt.xt) then
		call epictimc(xdat2,xtim2,itmax)
	    endif
	endif
	return
	end
