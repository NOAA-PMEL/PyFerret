	SUBROUTINE PLOTZ(Z,XF,YF,XLPLOT,YLPLOT,XHPLOT,YHPLOT)
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
C**     
C**     @(#)plotz.F	1.4    9/9/88
C**     
C**     
C***********************************************************************
C**     
C**     PLOT+ Scientific Graphics System
C**     
C***********************************************************************
C**     
C**     
C       lots of places ---> PPLMOD_PPLCMD ---> PPLMOD_PLTIT ---> PPLMOD_PLOTZ
C       
C       This subroutine is called for CONTOUR, SHADE and FILL plots
C       and puts the data on the proper grid and then plots it.
C 
C       Z      - actual 2D data array
C       XF     - X factor ( page? length / X range )
C       YF     - Y factor
C       XLPLOT - lower X page? coordinate
C       YLPLOT - lower Y page? coordinate
C       XHPLOT - upper X page? coordinate
C       YHPLOT - upper Y page? coordinate
C       
*       MOD *JD* to include AREA in TMAP PPL+
*       MOD *JD* moved dimension z(nx,ny) from below subroutine statement to
*       below includes so that CONTOUR will work on the MAC -- Ed Flinchem
*       had to do this to make that work. 3.29.93 DIMENSION --> REAL as well.
*       MOD *JD* To permit use of centered levels and specify the delta Z
*       value as well.  Added variables to CONT.INC common 8.3.95  
*	*jd* 8.28 fix bug in dz specification
*	*jd* 2.96 Change centered option so that data not spanning zero
*       is centered as well.
*       *jc* 4.97 Major clean up and commenting
C       *jc* 5.97 Added code for curvilinear coordinates

#ifdef unix
	INCLUDE 'pplv11inc/PARAMPL5.DAT'
	INCLUDE 'pplv11inc/ZGRIDD.INC'
	INCLUDE 'pplv11inc/ZGRID.INC'
	INCLUDE 'pplv11inc/AXIS.INC'
	include	'tmap_pplv11inc/cont_inc.decl'
	INCLUDE 'pplv11inc/CONT.INC'
	INCLUDE 'pplv11inc/HD.INC'
	INCLUDE 'pplv11inc/TAXIS.INC'
	INCLUDE 'pplv11inc/AXISL.INC'
	INCLUDE 'pplv11inc/PEN.INC'
	INCLUDE 'pplv11inc/LINES.INC'
	INCLUDE 'pplv11inc/PLTL.INC'
	include 'ferret_cmn/xcurvilinear.cmn'
#else
	INCLUDE 'pplinc:PARAMPL5.DAT'
	INCLUDE 'pplinc:ZGRIDD.INC'
	INCLUDE 'pplinc:ZGRID.INC'
	INCLUDE 'pplinc:AXIS.INC'
	include	'tmap_pplv11inc:cont_inc.decl'
	INCLUDE 'pplinc:CONT.INC'
	INCLUDE 'pplinc:HD.INC'
	INCLUDE 'pplinc:TAXIS.INC'
	INCLUDE 'pplinc:AXISL.INC'
	INCLUDE 'pplinc:PEN.INC'
	INCLUDE 'pplinc:LINES.INC'
	INCLUDE 'pplinc:PLTL.INC'
#endif
**	DIMENSION Z(NX,NY)
	REAL	  Z(NX,NY)

C*       *jd* Linux port 12.96 add EXTERNAL calls below
        external scale, range

C	VAX/VMS
C
	DIMENSION IZMRKD(3)
	DATA IZMRKD/12,16,3/
C
C	END VAX/VMS
C
C
C       IAUTOC = 1 means: compute levels from input data and number
C                         of levels

	IF (IAUTOC.EQ.1) THEN

	   IF (ZMIN.EQ.ZMAX) THEN

	      NLEV2=0

	   ELSE
       
C       If set_delta is .true. the delta value has been set
C       elsewhere and we should calculate the max and min values
C       of the range based on this delta.
C       (This portion stolen from RANGE)

	      if (set_delta) then 

		 dz = delta
C       
C       CALCULATE THE MINIMUM VALUE OF THE RANGE
C       
		 FM = zmin / DZ
		 M = IFIX(FM)
		 IF (FM .LT. 0.0) M = M - 1
		 znp = DZ * FLOAT(M)
C       
C       CALCULATE THE MAXIMUM VALUE OF THE RANGE.
C       
		 FM = zmax / DZ
		 M = IFIX(FM)
		 IF(FM .GT. 0.0)M = M + 1
		 zmp = DZ * FLOAT(M)

C       Else we should allow RANGE to determine the max and min range
C       values and also the appropriate delta value. (ZNP, ZMP, DZ)

	      else

		 CALL RANGE(ZMIN,ZMAX,NLEV,ZNP,ZMP,DZ)

	      endif

C       Set levels centered about zero if centered is .true.

	      if (centered) then
		 if (abs(znp) .gt. abs(zmp)) then
		    zmp = -znp
		 else
		    znp = -zmp
		 endif
	      endif

	      NLEV2=(ZMP-ZNP)/DZ+1

C       Handle rounding errors *jd* 6.27.95
C       JC_TODO This is where the rounding error for negative plots probably exists

	      IF (ZNP + DZ*(NLEV2-1) .LT. ZMP) NLEV2 = NLEV2 + 1

	      JJ=NSIGC-IFIX(ALOG10(ABS(ZMP-ZNP))+.5)
	      IF (JJ.LE.0) JJ=-1

C       ldig()  number of decimal places in the contour labels
C       =0  no decimal
C       =-2 no label (?perhaps -1?)
C       
C       lwgt()
C       =1 standard line
C       =2 heavy line
C       =3 dashed line

	      DO 100 I=1,NLEV2
		 LWGT(I)=2
		 LDIG(I)=-1
		 ZLEV(I)=ZNP+DZ*FLOAT(I-1)
		 IF (ABS(ZLEV(I)).LT.1.E-5*DZ) GOTO 100
		 IF (ZLEV(I).GT.0.) THEN
		    LWGT(I)=1
		 ELSE
		    LWGT(I)=3
		 END IF
		 LDIG(I)=JJ
 100	      CONTINUE

	   ENDIF

C       
C       Else IAUTOC != 1 means: use supplied levels
C       
	ELSE

	   IF (ZMIN.EQ.ZMAX) THEN
	      NLEV2=0
	   ELSE
	      NLEV2=NLEV
	   ENDIF

	END IF


C       DRAW CONTOUR
C
C       from ppl/include/PPL.INC:
C
C       JTYPE: = 2 2-d irregular rectangular grid
C              = 1 1-d set
C              = 0 2-d regular rectangular grid (integer)
C              =-1 2-d second component of a vector regular grid
C              =-2 2-d second component of a vector irregular grid
C
C
C       from ppl/include/ZGRID.INC:
C
C       controls for the plotting of the positions of ungridded x,y,z data
C
C       IZTYP: =0 no points drawn
C              =1 points drawn along the x axis
C              =2 points drawn along the y axis
C              =3 points drawn at each raw input value
C       
C       IZMRK: =0 use the default mark
C              =n use mark n to denote the location
C
C       IZLEN: number of points in the x,y,z triplets

C
C       If it's a 2-d irregular grid plot and points have been drawn
C
	if (abs(jtype).eq.2 .and. iztyp.ne.0) then

	   call markh(hmark(1))

	   if (izmrk.eq.0) then
	      imrk=izmrkd(iztyp)
	   else
	      imrk=izmrk
	   endif

	   if (iztyp.eq.1) then

	      do 200 i=1,nx
		 call trans(0,z(i,ny+1),ylo,xt,y)
		 call plot(xt,y,0,imrk)
		 call mark(imrk)
 200	      continue

	   else if (iztyp.eq.2) then

	      do 210 j=1,ny
		 call trans(0,xlo,z(j,ny+2),x,yt)
		 call plot(x,yt,0,imrk)
		 call mark(imrk)
 210	      continue

	   else

	      do 230 i=1,nx
		 do 220 j=1,ny
		    call trans(0,z(i,ny+1),z(j,ny+2),xt,yt)
		    call plot(xt,yt,0,imrk)
		    call mark(imrk)
 220		 continue
 230	      continue

	   endif

	endif

C
C       If there are some x,y,z triplets and points have been drawn
C
	IF (IZLEN.NE.0.AND.IZTYP.NE.0) THEN

	   CALL MARKH(HMARK(1))

	   IF (IZMRK.EQ.0) THEN
	      IMRK=IZMRKD(IZTYP)
	   ELSE
	      IMRK=IZMRK
	   ENDIF

	   IF (IZTYP.EQ.1) THEN

	      call trans(0,xx(1),ylo,xt,y)
	      CALL PLOT(XT,Y,0,IMRK)
	      CALL MARK(IMRK)
	      DO 300 I=2,IZLEN
		 IF (XT.NE.XX(I)) THEN
		    call trans(0,xx(i),ylo,xt,y)
		    CALL PLOT(xt,Y,0,IMRK)
		    CALL MARK(IMRK)
		 ENDIF
 300	      CONTINUE

	   ELSE IF (IZTYP.EQ.2) THEN

	      call trans(0,xlo,yy(1),x,yt)
	      CALL PLOT(X,YT,0,IMRK)
	      CALL MARK(IMRK)
	      DO 310 I=2,IZLEN
		 IF (YT.NE.YY(I)) THEN
		    call trans(0,xlo,yy(i),x,yt)
		    CALL PLOT(X,yt,0,IMRK)
		    CALL MARK(IMRK)
		 ENDIF
 310	      CONTINUE

	   ELSE

	      DO 320 I=1,IZLEN
		 call trans(0,xx(i),yy(i),xt,yt)
		 CALL PLOT(xt,yt,0,IMRK)
		 CALL MARK(IMRK)
 320	      CONTINUE

	   ENDIF

	ENDIF

	CALL SCALE(1.,1.,XORG,YORG,0.,0.)
	IF (IWIND.NE.0) CALL WINDOW(0.0,0.0,XLEN,YLEN)
	CALL COLOR(IPEN(1))


C       Calculate the following quantities:
C       ~FF - multiplication factors to go from axis to page coords?
C       ~OF - page offset to the beginning of an axis?

	call trans(0,xmin,ymin,xtm,ytm)
	call trans(0,xmax,ymax,xtx,ytx)
	XFF = (XHPLOT - XLPLOT)/(xtx-xtm)
	XOF = XLPLOT - XFF*xtm
	YFF = (YHPLOT - YLPLOT)/(ytx-ytm)
	YOF = YLPLOT - YFF*ytm


C       If this is a 2-D plot on an irregular grid, use TRANS() to 
C       scale the points appropriately.  Values are returned in 
C       the xx() and yy() arrays.
C
	IF (abs(jtype).eq.2) THEN
       
	   do 400 i=1,nx
	      call trans(0,z(i,ny+1),ymin,xx(izlen+i),yt)
 400	   continue
	   do 410 j=1,ny
	      call trans(0,xmin,z(j,ny+2),xt,yy(izlen+j))
 410	   continue


C       For all other plot types fill in XX() and YY() by
C       multiplying the index value by the DX or DY delta values.

	ELSE

	   DX = (xtx - xtm)/(NX - 1)
	   DY = (ytx - ytm)/(NY - 1)

	   DO 500 I=1,NX
	      XX(IZLEN+I)=DX*(I-1)+xtm
 500	   CONTINUE
	   DO 510 J=1,NY
	      YY(IZLEN+J)=DY*(J-1)+ytm
 510	   CONTINUE

	ENDIF


	 if (makep.eq.'CONTOUR') then
c *kob* 8/97 had to move the continuation marker to col 5 - it was in col 8
c       this bombed aix compiler, though it seems to work on others
	    CALL CONSEG(Z,NX,NY,XX(IZLEN+1),YY(IZLEN+1),NX,NY,
     1	 XFF,XOF,YFF,YOF,ZLEV,LDIG,LWGT,LCLR,NLEV2,HGT,
     2	 NARC,CANGLE)

	 else if (makep.eq.'FILL') then

	    call arfill(z,xx(izlen+1),yy(izlen+1),nx,ny,
     1	 zlev,nlev2)
C       CALL DAXIS(XF,YF,NMX,NMY)

	 else
*******     SQRFILL NOT A TMAP PPL+ COMMAND *jd* 2.19.99
*******	    call sqrfill(z,xx(izlen+1),yy(izlen+1),nx,ny,
*******     1	 zlev,nlev2)

	 endif

C
C	END VAX/VMS
C
	CALL COLOR(IPEN(0))
	CALL SCALE(XF,YF,XORG,YORG,XLO,YLO)

	RETURN
	END
