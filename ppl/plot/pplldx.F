	subroutine pplldx(icode,xt,yt,npts,tstrt,xdt)
C** 
C**    @(#)pplldx.F	1.2    9/8/88
C**
C**
C***********************************************************************
C**
C**                 PLOT+ Scientific Graphics System
C**
C***********************************************************************
C**
C**
c
c	icode	0 use both x and y
c		1 use x only
c		2 use y only
c	xt	x data
c	yt	y data
c	npts	number of xt and yt points
c	tstrt	start time, corresponds to a value of xt=1.0
c	xdt	sample rate in minutes for x
c
c	tstrt and xdt are used only for TAXIS,ON.
c
*............. I changed the logicals on the includes - no impact ...sh
#ifdef unix
	include 'pplinc/PARAMPL5.DAT'
	include 'pplinc/HD.INC'
	include 'pplinc/LINES.INC'
	include 'pplinc/DATA.INC'
	include 'pplinc/AXISL.INC'
	include 'pplinc/TAXIS.INC'
	include 'pplinc/CMRDL.INC'
	include 'pplinc/PPLDAT.INC'
#else
	include 'pplinc:parampl5.dat'
	include 'pplinc:hd.inc'
	include 'pplinc:lines.inc'
	include 'pplinc:data.inc'
	include 'pplinc:axisl.inc'
	include 'pplinc:taxis.inc'
	include 'pplinc:cmrdl.inc'
	include 'pplinc:ppldat.inc'
#endif
c
	dimension xt(*),yt(*),stop(3),temp(3),start(3)
	character tstrt*10
	integer*4 mtmin,tstart,tstop,time,tmin
c
	if(linen.eq.0)call rsmnmx
	linen=linen+1
	jtype=1
	nx=npts
	ny=1
	if(itflg.eq.1)then
c
c	time axis is on.
c
	    if(linen.eq.1)then
		iform='LDX'
		if(iautot.eq.1)itstrt=tstrt	
	    endif
c
c	calculate offset and rate for time data
c
	    off=diff(itstrt,tstrt)/dt+1.0
	    rate=xdt/dt
	else
	    off=1.0
	    rate=1.0
	endif
c
c	load data
c
	ix=ibase-1
	iy=ix+nsize/2
	if(icode.eq.0)then
	    do 100 i=1,npts
		x(iy+i)=yt(i)
		x(ix+i)=(xt(i)-1.0)*rate+off
100	    continue
	else if(icode.eq.1)then
	    do 200 i=1,npts
		x(iy+i)=i
		x(ix+i)=(xt(i)-1.0)*rate+off
200	    continue
	else
	    do 300 i=1,npts
		x(ix+i)=(float(i)-1.0)*rate+off
		x(iy+i)=yt(i)
300	    continue
	endif
	xmin=1.0e36
	xmax=-xmin
	ymin=xmin
	ymax=xmax
	do 400 i=1,npts
	if(xmin.gt.x(ix+i))xmin=x(ix+i)
	if(xmax.lt.x(ix+i))xmax=x(ix+i)
	if(ymin.gt.x(iy+i))ymin=x(iy+i)
	if(ymax.lt.x(iy+i))ymax=x(iy+i)
400	continue
c
c	some variable definitions
c
c	tmin    time for xmin
c	tstop   time for xmax
c	itmin   earliest tmin for all lines
c	itmax   latest  tstop for all lines
c	itstart reference start time (if iautot the first lines tstrt)
c
	if(itflg.eq.1)then
	    if(linen.eq.1)then
		read(itstrt,999)iy,im,id,ihr
999		format(3i2.2,i4.4)
		call mdymt(temp,im,id,iy)
		temp(3)=ihr
		tstart=mtmin(temp)
c
c	calc itmin
c
		tmin=tstart+(xmin-off)*dt
		call minmt(tmin,start)
		call mtmdy(start,im,id,iy)
		write(itmin,999)iy,im,id,ifix(start(3))
c
c	calc itmax
c
		tstop=tstart+(xmax-off)*dt
		call minmt(tstop,stop)
		call mtmdy(stop,im,id,iy)
		write(itmax,999)iy,im,id,ifix(stop(3))
	    else
		tmin=tstart+(xmin-off)*dt
		tstop=tstart+(xmax-off)*dt
		read(itmin,999)iy,im,id,ihr
		call mdymt(temp,im,id,iy)
		temp(3)=ihr
		time=mtmin(temp)
		if(tmin.lt.time)then
		    call minmt(tmin,start)
		    call mtmdy(start,im,id,iy)
		    write(itmin,999)iy,im,id,ifix(start(3))
		endif
c
		read(itmax,999)iy,im,id,ihr
		call mdymt(temp,im,id,iy)
		temp(3)=ihr
		time=mtmin(temp)
		if(tstop.gt.time)then
		    call minmt(tstop,stop)
		    call mtmdy(stop,im,id,iy)
		    write(itmax,999)iy,im,id,ifix(stop(3))
		endif
	    endif
	endif
	lleng(linen)=npts
	pbuf=1
	call stmnmx(x,nsize,xmin,xmax,ymin,ymax,0.,0.)
	lnum = linen
	if(linen.ge.NLINES)then
	    linen=NLINES - 1
	else
	    ibase=ibase+lleng(linen)
	endif
	return
	end
