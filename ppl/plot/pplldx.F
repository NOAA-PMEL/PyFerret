      subroutine pplldx(icode,xt,yt,npts,tstrt,tref,xdt,pplmem)


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
c
c                  When saving polygon data, save npts per polygon with 
c                  the data, not in the array lleng(): unlimited # polygons

c               3 use both x and y, for polygons 
c		4 use x only, for polygons
c		5 use y only, for polygons
c
c	xt	x data
c	yt	y data
c	npts	number of xt and yt points
c	tstrt	start time, corresponds to a value of xt=1.0
c 	tref	added 11.95, a reference time (~tstart) ferret will set
c	xdt	sample rate in minutes for x
c	pplmem  data buffer for PPLUS data
c
c	tstrt and xdt are used only for TAXIS,ON.
c
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
* V510 *sh* 3/00 - allow loading of up to NPOLYGON lines (instead of NLINES)
*		 - the X or Y inputs no longer both flagged as bad if either is
* V540 *acm*9/01 - changes for dynamic pplus buffer, call PPLLDX via pplldx_envelope.c
*                  which checks pplmem_nsize (new name for NSIZE) and allocates 
*                  more if needed.
*      *acm* 10/01 Remove VMS includes
* V541 *acm*  4/02 Explicitly set itstart to be all 14 character places of tref:
*                  itstrt = tref(1:14) Otherwise we only get the first 10 (why??)
* V553 *acm*  6/03  For polygons, store the # of points in each polygon in the PPLUS
*                   memory buffer, rather than in lleng.  This way the number of
*                   polygons that may be plotted is unlimited. Mark this with the
*                   value of icode (first argument to pplldx_envelope)
* V68  *acm* 1/12  changes for double-precision ferret, single-precision pplus
*      *kms* 5/12 Always use generic REAL instead of specific SNGL 
*                 (SNGL must have a double-precision or complex argument)

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

	include 'parampl5_dat.decl'
	include 'PARAMPL5.DAT'
	include 'hd_inc.decl'
	include 'HD.INC'
	include 'lines_inc.decl'
	include 'LINES.INC'
	include 'taxis_inc.decl'
	include 'TAXIS.INC'
	include 'cmrdl_inc.decl'
	include 'CMRDL.INC'
	include 'data_inc.decl'
	include 'DATA.INC'
	include 'axisl_inc.decl'
	include 'AXISL.INC'
	include 'PPLDAT.INC'

	include 'miss_inc.decl'
	include 'MISS.INC'

* 	Add external stmt for linux port
	external range


CCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCC

c
! original:
!	dimension xt(*),yt(*),stop(3),temp(3),start(3)
!	character tstrt*10
!	integer*4 mtmin,tstart,tstop,time,tmin

        REAL pplmem(*)

#ifdef double_p
	REAL*8 xt(*),yt(*), xdt
#else
	REAL xt(*),yt(*), xdt
#endif
	CHARACTER BC2WHOI*14, tstrt*(*), tref*(*)
	REAL*8	WHOI2BC, DIFF, tstart, tstop, time, tmin, off, rate, 
     .		dxmin, dxmax, dymin, dymax, xymin, xymax dtd
	INTEGER	icode, npts, i, ix, iy
	REAL	xs, ys
	LOGICAL  TM_FPEQ_SNGL
	LOGICAL  flip			!  FERRET mod 10/93

	REAL	    bad_flag
	PARAMETER ( bad_flag = -3.E33 )

	logical	translate 

* end of 9-21 FERRET declaration changes

* mod to SAVE certain variables *jd* 6.5.91
        SAVE        tstart,tmin,tstop

c
	dtd = dt

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
c ** acm mod 9/01  Compare tref(1:1) rather than just tref. After
c **               passing thru c routine, a blank tref is somehow different??

	    if(linen.eq.1)then
		iform='LDX'
		if(iautot.eq.1 .AND. tref(1:1).NE.' ') itstrt=tref(1:14) 
	    endif		
c
c	calculate offset and rate for time data
c
	    off=diff(itstrt,tstrt)/dtd+1.0
	    rate=xdt/dtd
**	else
**	    off=1.0
**	    rate=1.0
	endif
c
c	load data
c

C 8/2015 compute min and max in double precision, then translate

	dxmin=1.0e36
	dxmax=-dxmin
	dymin=dxmin
	dymax=dxmax

	ix=ibase-1
	iy=ix+pplmem_nsize/2
	if(icode.eq.0 .OR. icode.EQ.3) THEN
	    do 100 i=1,npts

	        xs = REAL(xt(i))
	        ys = REAL(yt(i))
	        IF ( flip ) THEN

	          IF ( yeq .AND. TM_FPEQ_SNGL(ys,cmyeq) ) THEN
	             pplmem(iy+i) = bad_flag

		  elseif (translate) then
		     pplmem(iy+i)=(yt(i)-1.0)*rate+off
		  else
		     pplmem(iy+i)=yt(i)
		  endif

	          IF ( xeq .AND. TM_FPEQ_SNGL(xs,cmxeq)) THEN
	             pplmem(ix+i) = bad_flag
	          ELSE
		     pplmem(ix+i)=xt(i)
	          ENDIF

	        ELSE
	          IF ( yeq .AND. TM_FPEQ_SNGL(ys,cmyeq) ) THEN
	             pplmem(iy+i) = bad_flag
	          ELSE
		     pplmem(iy+i)=yt(i)
	          ENDIF

	          IF ( xeq .AND. TM_FPEQ_SNGL(xs,cmxeq) ) THEN
	             pplmem(ix+i) = bad_flag
	          ELSEIF (translate) then
		     pplmem(ix+i)=(xt(i)-1.0)*rate+off
	          ELSE
		     pplmem(ix+i)=xt(i)
		  ENDIF

		  IF ( .NOT.(xeq .AND. TM_FPEQ_SNGL(xs,cmxeq)) ) THEN
		     dxmin = MIN(dxmin, xt(i))
		     dxmax = MAX(dxmax, xt(i))
		  ENDIF
		  IF ( .NOT.(yeq .AND. TM_FPEQ_SNGL(ys,cmyeq)) ) THEN
		     dymin = MIN(dymin, yt(i))
		     dymax = MAX(dymax, yt(i))
		  ENDIF

		ENDIF

100	    continue

            if (flip) THEN
	       IF (translate) then
		   dymin = (dymin-1.0)*rate+off
		   dymax = (dymax-1.0)*rate+off
	       ENDIF
	    ELSE
	       IF (translate) then
		   dxmin = (dxmin-1.0)*rate+off
		   dxmax = (dxmax-1.0)*rate+off
	       ENDIF
	    ENDIF

	else if(icode.eq.1)then
	    do 200 i=1,npts
	        xs = REAL(xt(i))
	        IF ( ( xeq .AND. TM_FPEQ_SNGL(xs,cmxeq) ) ) THEN
	           pplmem(ix+i) = bad_flag
	           pplmem(iy+i) = i     ! 3/00 mod
	           GOTO 200
	        ENDIF
		pplmem(iy+i)=i
	        if (translate) then
		   pplmem(ix+i)=(xt(i)-1.0)*rate+off
	        else
		   pplmem(ix+i)=xt(i)
	        endif
200	    continue
	else
	    do 300 i=1,npts
	        ys = REAL(yt(i))
	        IF ( ( yeq .AND. TM_FPEQ_SNGL(ys,cmyeq) ) ) THEN
	           pplmem(ix+i) = bad_flag
	           pplmem(iy+i) = bad_flag
	           GOTO 300
	        ENDIF
		if (translate) then
		   pplmem(ix+i)=(float(i)-1.0)*rate+off
	        else
		   pplmem(ix+i)=float(i)
	        endif
		pplmem(iy+i)=yt(i)
300	    continue
	endif
* FERRET mod 2/89 - replace bad data LIMITS flags
	cmxeq = bad_flag
	cmyeq = bad_flag
c
c	some variable definitions
c
c	tmin    time for xmin (endpoint of this line in minutes since BC)
c	tstop   time for xmax (endpoint of this line in minutes since BC)
c	itmin   earliest tmin for all lines (in WHOI format)
c	itmax   latest  tstop for all lines (in WHOI format)
c	itstart reference start time (if iautot the first lines tstrt) (???)
c

	if(translate)then  		! 11.30.95 WAS:	if(itflg.eq.1)then
		if(itflg.eq.1) then 	! 11.30.95 for both x and y
		   xymin  = dxmin
		   xymax  = dxmax
	        else 			! gotta be -1
		   xymin  = dymin
		   xymax  = dymax
		endif

	    if(linen.eq.1)then
		tstart = WHOI2BC( itstrt )		! t=1 of time axis

		tmin=tstart+(xymin-1.0)*dtd		! start of plot axis
		itmin = BC2WHOI( tmin )

		tstop=tstart+(xymax-1.0)*dtd		! end of plot axis
		itmax = BC2WHOI( tstop )

	    else

		tmin=tstart+(xymin-1.0)*dtd
		tstop=tstart+(xymax-1.0)*dtd

		time = WHOI2BC( itmin )
		if(tmin.lt.time)  itmin = BC2WHOI( tmin )   ! chg. plot start
c
		time = WHOI2BC( itmax )
		if(tstop.gt.time) itmax = BC2WHOI( tstop )   ! chg. plot end
	    endif
	endif
* end of FERRET 9-21 changes
        
	pbuf=1
c 8/2015 Should stmnmx use double precisison xmin,xmax, ...?  Seems ok as single.
	xmin = dxmin
	xmax = dxmax
	ymin = dymin
	ymax = dymax
	call stmnmx(pplmem,pplmem_nsize,xmin,xmax,ymin,ymax)

	lnum = linen
        IF (icode .LE. 2) THEN
           lleng(linen)=npts
 	   if(linen.ge.NPOLYGON)then
	       linen=NPOLYGON - 1
	   else
	       ibase=ibase+lleng(linen)
	   endif
        ELSE
           ibase = ibase + npts
        ENDIF

	return
	end
