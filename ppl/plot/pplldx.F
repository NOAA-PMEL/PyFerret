	subroutine pplldx(icode,xt,yt,npts,tstrt,tref,xdt)
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
C *kob* 04/07/96 passed in character strings had to be declared 
C                with "*(*)" for the linux port 
c
c	icode	0 use both x and y
c		1 use x only
c		2 use y only
c	xt	x data
c	yt	y data
c	npts	number of xt and yt points
c	tstrt	start time, corresponds to a value of xt=1.0
c 	tref	added 11.95, a reference time (~tstart) ferret will set
c	xdt	sample rate in minutes for x
c
c	tstrt and xdt are used only for TAXIS,ON.
c
*............. I changed the logicals on the includes - no impact ...sh
* revised for FERRET
* from PPLUS :

*	Directory PMEL:[PLOT5.LIBRARY.PLOT]

*	PPLLDX.FOR;1              7  25-AUG-1987 15:24

* search for "FERRET" to find all revisions
* *sh* 9/21/88 - rev 1.0 for FERRET 1.10+
*	- extensive changes to code that determines time axis limits
*	  previous version was based on a "minutes since 1970" calculation
*	  that blew up on dates preceeding 1970.  Replaced with a "minutes
*	  since BC" calculation
* *sh* 2/20/89- rev 1.1 for FERRET 1.20
*	- corrected bug in FERRET code: tmin,tstop calculation for linen>1
*	- reference ALL time axes to the same T0='0101010000' date
*	  instead of using T0 from first line (else overlays don't work)
*       - added LIMITS screening - permitting different bad data indicators
*	  in each data set
* Note: calling PPLLDX modifies the XEQ and YEQ limits
*	Mod *jd* 12.23.92 for MAC FERRET to change 8 args to 6 in stmnmx call
* *sh* 10/25/93 - for version 3.01 of FERRET
*	- corrected bug if Y axis was the time axis - data was not put through
*	  the time/date translation pipeline
* *sh*jd* 7.31.95 
*	- Fixes bug where small axis ranges result in round off error 
*	  creation.  Use TRANSLATE to catch.
* *jd* 11.27.95 Mod for 4 digit year
* *jd* 11.28.95 Add additional arg "tref" so reference time can be set
*	        by Ferret, now that 1901 is insufficient
* *jd* 11.30.95 Fix bug (again? see 10/25/93) where Y axis time axis
*		limits are not computed

* FERRET 9-21-88: replaced include files to make all declarations explicit
! original:

!	include 'pplinc:parampl5.dat'
!	include 'pplinc:hd.inc'
!	include 'pplinc:lines.inc'
!	include 'pplinc:data.inc'
!	include 'pplinc:axisl.inc'
!	include 'pplinc:taxis.inc'
!	include 'pplinc:cmrdl.inc'
!	include 'pplinc:ppldat.inc'

#ifdef unix
	include 'tmap_pplv11inc/parampl5_dat.decl'
	include 'pplv11inc/PARAMPL5.DAT'
	include 'tmap_pplv11inc/hd_inc.decl'
	include 'pplv11inc/HD.INC'
	include 'tmap_pplv11inc/lines_inc.decl'
	include 'pplv11inc/LINES.INC'
	include 'tmap_pplv11inc/taxis_inc.decl'
	include 'pplv11inc/TAXIS.INC'
	include 'tmap_pplv11inc/cmrdl_inc.decl'
	include 'pplv11inc/CMRDL.INC'
	include 'tmap_pplv11inc/data_inc.decl'
	include 'pplv11inc/DATA.INC'
	include 'tmap_pplv11inc/axisl_inc.decl'
	include 'pplv11inc/AXISL.INC'
	include 'tmap_pplv11inc/ppldat_inc.decl'
	include 'pplv11inc/PPLDAT.INC'
#else
	INCLUDE 'PPLV11INC:parampl5.dat'
	INCLUDE 'PPLV11INC:hd.inc'
	INCLUDE 'PPLV11INC:lines.inc'
	INCLUDE 'PPLV11INC:taxis.inc'
	INCLUDE 'PPLV11INC:cmrdl.inc'
	INCLUDE 'PPLV11INC:data.inc'
	INCLUDE 'PPLV11INC:axisl.inc'
	INCLUDE 'PPLV11INC:ppldat.inc'
#endif

#ifdef unix
	include 'tmap_pplv11inc/miss_inc.decl'
	include 'pplv11inc/MISS.INC'
#else
	INCLUDE 'PPLV11INC:miss.inc'
#endif

* 	Add external stmt for linux port
	external range


CCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCC

c
! original:
!	dimension xt(*),yt(*),stop(3),temp(3),start(3)
!	character tstrt*10
!	integer*4 mtmin,tstart,tstop,time,tmin

	REAL xt(*),yt(*)
	CHARACTER BC2WHOI*14, tstrt*(*), tref*(*)
	REAL*8 	  WHOI2BC, tstart,tstop,time,tmin
	INTEGER	icode, npts, i, ix, iy
	REAL	DIFF, xdt, off, rate
	LOGICAL  flip			!  FERRET mod 10/93

	REAL	    bad_flag
	PARAMETER ( bad_flag = -3.E33 )

	logical	translate 

* end of 9-21 FERRET declaration changes

* mod to SAVE certain variables *jd* 6.5.91
        SAVE        tstart,tmin,tstop

c
	if(linen.eq.0)call rsmnmx
	linen=linen+1
	jtype=1
	nx=npts
	ny=1
	flip = itflg .EQ. -1		!  FERRET mod 10/93



**	if(ABS(itflg).eq.1)then		!  FERRET mod 10/93  "ABS"

	translate = ABS(itflg).eq.1
	if (translate) then	
c
c	time axis is on.
c
	    if(linen.eq.1)then
		iform='LDX'
		if(iautot.eq.1 .AND. tref.NE.' ')itstrt=tref 
	    endif		
c
c	calculate offset and rate for time data
c
	    off=diff(itstrt,tstrt)/dt+1.0
	    rate=xdt/dt
**	else
**	    off=1.0
**	    rate=1.0
	endif
c
c	load data
c
	ix=ibase-1
	iy=ix+nsize/2
	if(icode.eq.0)then
	    do 100 i=1,npts
* FERRET mod 2/89 - screen bad data
	        IF ( ( xeq .AND. xt(i).EQ.cmxeq )
     .	        .OR. ( yeq .AND. yt(i).EQ.cmyeq ) ) THEN
	           x(ix+i) = bad_flag
	           x(iy+i) = bad_flag
	           GOTO 100
	        ENDIF
* end of 2/89 FERRET mod
* FERRET mod 10/93 - apply transformation to Y data
	        IF ( flip ) THEN

		  if (translate) then
		    x(iy+i)=(yt(i)-1.0)*rate+off
		  else
		    x(iy+i)=yt(i)
		  endif

		  x(ix+i)=xt(i)
	        ELSE
		  x(iy+i)=yt(i)
		  if (translate) then
		     x(ix+i)=(xt(i)-1.0)*rate+off
	          else
		     x(ix+i)=xt(i)
		  endif

		ENDIF
* end of 10/93 FERRET mod
100	    continue
	else if(icode.eq.1)then
	    do 200 i=1,npts
* FERRET mod 2/89 - screen bad data
	        IF ( ( xeq .AND. xt(i).EQ.cmxeq ) ) THEN
	           x(ix+i) = bad_flag
	           x(iy+i) = bad_flag
	           GOTO 200
	        ENDIF
* end of 2/89 FERRET mod
		x(iy+i)=i
	        if (translate) then
		   x(ix+i)=(xt(i)-1.0)*rate+off
	        else
		   x(ix+i)=xt(i)
	        endif
200	    continue
	else
	    do 300 i=1,npts
* FERRET mod 2/89 - screen bad data
	        IF ( ( yeq .AND. yt(i).EQ.cmyeq ) ) THEN
	           x(ix+i) = bad_flag
	           x(iy+i) = bad_flag
	           GOTO 300
	        ENDIF
* end of 2/89 FERRET mod
		if (translate) then
		   x(ix+i)=(float(i)-1.0)*rate+off
	        else
		   x(ix+i)=float(i)
	        endif
		x(iy+i)=yt(i)
300	    continue
	endif
* FERRET mod 2/89 - replace bad data LIMITS flags
	cmxeq = bad_flag
	cmyeq = bad_flag
* end of 2/89 FERRET mod
	xmin=1.0e36
	xmax=-xmin
	ymin=xmin
	ymax=xmax
	do 400 i=1,npts
* FERRET mod 2/89 - eliminate bad data from min/max calculation
	IF ( x(ix+i).EQ.bad_flag .OR. x(iy+i).EQ.bad_flag ) GOTO 400
* end of 2/89 FERRET mod
	if(xmin.gt.x(ix+i))xmin=x(ix+i)
	if(xmax.lt.x(ix+i))xmax=x(ix+i)
	if(ymin.gt.x(iy+i))ymin=x(iy+i)
	if(ymax.lt.x(iy+i))ymax=x(iy+i)
400	continue
c
c	some variable definitions
c
c	tmin    time for xmin (endpoint of this line in minutes since BC)
c	tstop   time for xmax (endpoint of this line in minutes since BC)
c	itmin   earliest tmin for all lines (in WHOI format)
c	itmax   latest  tstop for all lines (in WHOI format)
c	itstart reference start time (if iautot the first lines tstrt) (???)
c
* FERRET 9-21-88: replaced code dependent on 1970 reference
! original:
!	if(itflg.eq.1)then
!	    if(linen.eq.1)then
!		read(itstrt,999)iy,im,id,ihr
!999		format(3i2.2,i4.4)
!		call mdymt(temp,im,id,iy)
!		temp(3)=ihr
!		tstart=mtmin(temp)
!c
!c	calc itmin
!c
!		tmin=tstart+(xmin-off)*dt
!		call minmt(tmin,start)
!		call mtmdy(start,im,id,iy)
!		write(itmin,999)iy,im,id,ifix(start(3))
!c
!c	calc itmax
!c
!		tstop=tstart+(xmax-off)*dt
!		call minmt(tstop,stop)
!		call mtmdy(stop,im,id,iy)
!		write(itmax,999)iy,im,id,ifix(stop(3))
!	    else
!		tmin=tstart+(xmin-off)*dt
!		tstop=tstart+(xmax-off)*dt
!		read(itmin,999)iy,im,id,ihr
!		call mdymt(temp,im,id,iy)
!		temp(3)=ihr
!		time=mtmin(temp)
!		if(tmin.lt.time)then
!		    call minmt(tmin,start)
!		    call mtmdy(start,im,id,iy)
!		    write(itmin,999)iy,im,id,ifix(start(3))
!		endif
!c
!		read(itmax,999)iy,im,id,ihr
!		call mdymt(temp,im,id,iy)
!		temp(3)=ihr
!		time=mtmin(temp)
!		if(tstop.gt.time)then
!		    call minmt(tstop,stop)
!		    call mtmdy(stop,im,id,iy)
!		    write(itmax,999)iy,im,id,ifix(stop(3))
!		endif
!	    endif
!	endif

	if(translate)then  		! 11.30.95 WAS:	if(itflg.eq.1)then
		if(itflg.eq.1) then 	! 11.30.95 for both x and y
		   xymin  = xmin
		   xymax  = xmax
	        else 			! gotta be -1
		   xymin  = ymin
		   xymax  = ymax
		endif

	    if(linen.eq.1)then
		tstart = WHOI2BC( itstrt )		! t=1 of time axis

		tmin=tstart+(xymin-1.0)*dt		! start of plot axis
		itmin = BC2WHOI( tmin )

		tstop=tstart+(xymax-1.0)*dt		! end of plot axis
		itmax = BC2WHOI( tstop )

	    else

		tmin=tstart+(xymin-1.0)*dt
		tstop=tstart+(xymax-1.0)*dt

		time = WHOI2BC( itmin )
		if(tmin.lt.time)  itmin = BC2WHOI( tmin )   ! chg. plot start
c
		time = WHOI2BC( itmax )
		if(tstop.gt.time) itmax = BC2WHOI( tstop )   ! chg. plot end
	    endif
	endif
* end of FERRET 9-21 changes

	lleng(linen)=npts
	pbuf=1
* ORIG	call stmnmx(x,nsize,xmin,xmax,ymin,ymax,0.,0.)
	call stmnmx(x,nsize,xmin,xmax,ymin,ymax)
	lnum = linen
	if(linen.ge.NLINES)then
	    linen=NLINES - 1
	else
	    ibase=ibase+lleng(linen)
	endif
	return
	end
