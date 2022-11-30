	subroutine plotny(xp,yp,ics,itype) 
*
*
*  This software was developed by the Thermal Modeling and Analysis
*  Project(TMAP) of the National Oceanographic and Atmospheric
*  Administration's (NOAA) Pacific Marine Environmental Lab(PMEL),
*  hereafter referred to as NOAA/PMEL/TMAP.
*
*  Access and use of this software shall impose the following
*  obligations and understandings on the user. The user is granted the
*  right, without any fee or cost, to use, copy, modify, alter, enhance
*  and distribute this software, and any derivative works thereof, and
*  its supporting documentation for any purpose whatsoever, provided
*  that this entire notice appears in all copies of the software,
*  derivative works and supporting documentation.  Further, the user
*  agrees to credit NOAA/PMEL/TMAP in any publications that result from
*  the use of this software or in any product that includes this
*  software. The names TMAP, NOAA and/or PMEL, however, may not be used
*  in any advertising or publicity to endorse or promote any products
*  or commercial entity unless specific written permission is obtained
*  from NOAA/PMEL/TMAP. The user also understands that NOAA/PMEL/TMAP
*  is not obligated to provide the user with any support, consulting,
*  training or assistance of any kind with regard to the use, operation
*  and performance of this software nor to provide the user with any
*  updates, revisions, new versions or "bug fixes".
*
*  THIS SOFTWARE IS PROVIDED BY NOAA/PMEL/TMAP "AS IS" AND ANY EXPRESS
*  OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED
*  WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
*  ARE DISCLAIMED. IN NO EVENT SHALL NOAA/PMEL/TMAP BE LIABLE FOR ANY SPECIAL,
*  INDIRECT OR CONSEQUENTIAL DAMAGES OR ANY DAMAGES WHATSOEVER
*  RESULTING FROM LOSS OF USE, DATA OR PROFITS, WHETHER IN AN ACTION OF
*  CONTRACT, NEGLIGENCE OR OTHER TORTUOUS ACTION, ARISING OUT OF OR IN
*  CONNECTION WITH THE ACCESS, USE OR PERFORMANCE OF THIS SOFTWARE. 
*
*
C**    @(#)plotny.F	1.1    3/10/88
C**
C**
C***********************************************************************
C**
C**                 PLOT+ Scientific Graphics System
C**
C***********************************************************************
C**
C**
c     plots a line from present point to point xb,yb (in inches) 
c     ics=3 pen up 
c        =2 pen down 
c     itype=0 or 1 ordinary line 
c          =2 heavy line 
c          =3 dotted line 
c     must be initialized by call to plotny with pen up. 
c     oceanography/emr   december 1969. 
c     dotted line routine improved july/74 
c 
	common/rotcon/xk,yk,snthta,csthta
	include 'DASHZZ.INC'
c	data dashln/.02/ , spacln/.04/ , dslab/5.0/

	totlng=dashln+spacln 
	xb=xp*csthta-yp*snthta+xk
	yb=yp*csthta+xp*snthta+yk

	if(ics .eq. 3)then
	   call plot(xb,yb,3-ics,0) 
	   go to 100 
	endif

	if(itype .lt. 2) then
c 
c     ordinary line   itype=0 or 1. 
c 
	   call plot(xb,yb,3-ics,0) 
	   go to 100 

	else if (itype .eq. 2) then
c 
c     heavy line   itype=2 
c 
	   dx=xb-xa 
	   dy=yb-ya 
	   ds=sqrt(dx*dx+dy*dy)+.001 
	   dxd=dx*.01/ds 
	   dyd=dy*.01/ds 
	   call plot(xb,yb,1,0) 
	   call plot(xb+dyd,yb-dxd,1,0) 
	   call plot(xa+dyd,ya-dxd,1,0) 
	   call plot(xa-dyd,ya+dxd,1,0) 
	   call plot(xb-dyd,yb+dxd,1,0) 
	   call plot(xb,yb,1,0) 
	   go to 100 

	else

c 
c     dotted line   itype=3 
c 
30	   sx=xb-xa 
	   sy=yb-ya 
	   sabsq=sx**2+sy**2 
	   if(sabsq.eq.0.) go to 100 

	   sab=sqrt(sabsq) 
	   cose=sx/sab 
	   sine=sy/sab 
	   if(stlng.ge.dashln) then 
	      x=xa-(stlng-dashln)*cose 
	      y=ya-(stlng-dashln)*sine 
	      go to 50 
	   endif

	   x=xa-stlng*cose 
	   y=ya-stlng*sine 

40	   continue

	   x=x+dashln*cose 
	   y=y+dashln*sine 
	   ssq=(x-xa)**2+(y-ya)**2 

	   if(ssq.ge.sabsq) then 
	      call plot(xb,yb,1,0) 
	      stlng=dashln-(sqrt(ssq)-sab) 
	      go to 100 
	   endif

	   call plot(x,y,1,0) 

50	   continue

	   x=x+spacln*cose 
	   y=y+spacln*sine 
	   ssq=(x-xa)**2+(y-ya)**2 
	   if(ssq.ge.sabsq) then
	      call plot(xb,yb,0,0) 
	      stlng=totlng-(sqrt(ssq)-sab) 
	      go to 100 
	   endif

	   call plot(x,y,0,0) 
	   go to 40 

	endif

c 
c     save point a for next time 
c 
100	xa=xb 
	ya=yb 
	if(ics.eq.3) stlng=dashln/2.0 
	return 
	end 
