	subroutine epictimc (xdate,xtime,icom)
C** 
C**    @(#)epictimc.f	1.1    3/10/88
C**
C**
C***********************************************************************        C**
C**                 EPIC
C**
C***********************************************************************        C**
C**
c
c  Takes real numbers xdate,xtime (EPIC date/time) and writes
c  character date date/time in WHOI format (yymmddhhmm) for PLOT5.
c
c  Programmed by N. Soreide, May 86.
c
C *kob* 04/07/96 passed in character strings had to be declared 
C                with "*(*)" for the linux port 

	character icom*(*)
c
	write (icom,100) ifix(xdate+.5),ifix(xtime+.5)
100	format (i6.6,i4.4)
	return
	end
