	subroutine plotny(xp,yp,ics,itype) 
C** 
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
#ifdef unix
	include 'pplinc/DASHZZ.INC'
#else
	include 'pplinc:dashzz.inc'
#endif
c	data dashln/.02/ , spacln/.04/ , dslab/5.0/
	totlng=dashln+spacln 
	xb=xp*csthta-yp*snthta+xk
	yb=yp*csthta+xp*snthta+yk
	if(ics-3)5,10,5 
5	if(itype-2)10,20,30 
c 
c     ordinary line   itype=0 or 1. 
c 
10	call plot(xb,yb,3-ics,0) 
	go to 100 
c 
c     heavy line   itype=2 
c 
20	dx=xb-xa 
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
c 
c     dotted line   itype=3 
c 
30	sx=xb-xa 
	sy=yb-ya 
	sabsq=sx**2+sy**2 
	if(sabsq.eq.0.) go to 100 
	sab=sqrt(sabsq) 
	cose=sx/sab 
	sine=sy/sab 
	if(stlng.lt.dashln) go to 35 
	x=xa-(stlng-dashln)*cose 
	y=ya-(stlng-dashln)*sine 
	go to 50 
35	x=xa-stlng*cose 
	y=ya-stlng*sine 
40	x=x+dashln*cose 
	y=y+dashln*sine 
	ssq=(x-xa)**2+(y-ya)**2 
	if(ssq.lt.sabsq) go to 45 
	call plot(xb,yb,1,0) 
	stlng=dashln-(sqrt(ssq)-sab) 
	go to 100 
45	call plot(x,y,1,0) 
50	x=x+spacln*cose 
	y=y+spacln*sine 
	ssq=(x-xa)**2+(y-ya)**2 
	if(ssq.lt.sabsq) go to 55 
	call plot(xb,yb,0,0) 
	stlng=totlng-(sqrt(ssq)-sab) 
	go to 100 
55	call plot(x,y,0,0) 
	go to 40 
c 
c     save point a for next time 
c 
100	xa=xb 
	ya=yb 
	if(ics.eq.3) stlng=dashln/2.0 
	return 
	end 
