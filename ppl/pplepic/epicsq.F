	subroutine epicsq (xd,xt,iseq)
C** 
C**    @(#)epicsq.f	1.1    3/10/88
C**
C**
C***********************************************************************        C**
C**                 EPIC
C**
C***********************************************************************        C**
C**
c
c  Input:
c  xd,xt = EPIC date, time words (type REAL) (xd=yymmdd and xt=hhmm).
c
c  Output:
c  iseq = Sequence number in minutes from a date which is approximately
c         the start of the century.  (type INTEGER*4)
c
c  Uses MILELIB function JDAY to perform calculations.  Used to get
c  EPIC time words into a sequence number for PPLUS.
c
c  Programmed by N. Soreide, Dec 86.
c
	integer*4 jday,iseq
c
	idate=xd+.001
	itime=xt+.001
c
	iy=idate/10000
	im1=idate-iy*10000
	im=im1/100
	id=im1-im*100
	ihr=itime/100
	imin=itime-ihr*100
c
	iseq = jday (iy, im, id, ihr, imin)
c
	return
	end

