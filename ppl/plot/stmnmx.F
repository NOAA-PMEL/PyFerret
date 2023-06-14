	SUBROUTINE STMNMX(X,NSIZE,XMIN1,XMAX1,YMIN1,YMAX1)

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
* revised for TMAP
* from PPLUS :
*	Directory PMEL:[PLOT5.PPLV11.PLOT]MINMAX.FOR

* search for "TMAP" to find all revisions
* *sh* - 1/27/89 corrected bug: min/max search improperly rejects LIMITs points
* *sh* - 12/8/93 corrected bug: vectors blow on divied by zero attempt
* *jd* - 4.20.99 Don't delete symbols {X,Y}{MIN,MAX} in overlay plots
* V580 *acm* 8/04  Allow for PPL POLYGON as well as PPL FILLPOL after a POLY/SET command.
* V686 *acm* 11/13 Allow symbol names to be up to 120 characters long
* v695 9/15 *acm* write values to PPL* symbols using 6 digits not 4
* 11/2022 *acm* Code cleanup: Modernize DO-loop syntax.

	DIMENSION X(NSIZE)
C**
C**    @(#)minmax.F	1.1    3/10/88
C**
C**
C***********************************************************************
C**
C**                 PLOT+ Scientific Graphics System
C**
C***********************************************************************
C**
C**
        include 'parampl5_dat.decl'
        include 'PARAMPL5.DAT'
        include 'lines_inc.decl'
        include 'LINES.INC'
        include 'data_inc.decl'
        include 'DATA.INC'
        include 'miss_inc.decl'
        include 'MISS.INC'
	include 'hd_inc.decl'
	include 'HD.INC'
        include 'switch_inc.decl'
        include 'SWITCH.INC'
        include 'pltl_inc.decl'
        include 'PLTL.INC'

	CHARACTER SYM*120,VAL*10
C
	SYM='*PPL$LINE_COUNT'
	WRITE(VAL,'(I2)')LINEN
	CALL PUTSYM(SYM,VAL,2,IER)
C
	IF(LINEN.EQ.0)THEN
C
C	CONTOUR FORM
C
*jd*begin
	    IF (.NOT.SOVER) THEN
	       SYM='*PPL$XMIN'
	       CALL PUTVAL(SYM,XMIN1,7,IER)
C       
	       SYM='*PPL$XMAX'
	       CALL PUTVAL(SYM,XMAX1,7,IER)
C       
	       SYM='*PPL$YMIN'
	       CALL PUTVAL(SYM,YMIN1,7,IER)
C       
	       SYM='*PPL$YMAX'
	       CALL PUTVAL(SYM,YMAX1,7,IER)
	    ENDIF
*jd*end

c	find the min and max and the locations of them
c
	    ZMIN=1.E38
            ZMAX=-ZMIN
	    jmn=1
	    jmx=1
	    imx=1
	    imn=1
	    if(jtype.eq.-1)then
		ivlen=nx*ny
	    else
		ivlen=nx*ny+nx+ny
	    endif
            DO 301 I=1,NX
            DO 300 J=1,NY
		if(jtype.ge.0)then
* TMAP mods begin:
* note: a major bug still exists here: X and Y screening set by LIMITS command
*	are ignored
*  LIMITS screening test for jtype.ge.0 moved from below
                    XT=X(NX*(J-1)+I)
        	    IF(ZLE.AND.XT.LE.CMZLE)GOTO 300
        	    IF(ZEQ.AND.XT.EQ.CMZEQ)GOTO 300
        	    IF(ZGE.AND.XT.GE.CMZGE)GOTO 300
		else
                    xtt1=x(nx*(j-1)+i)
                    xtt2=x(ivlen+nx*(j-1)+i)
*  ... LIMITS screening test for vector plots modified
        	    IF(ZLE.AND.xtt1.LE.CMZLE)GOTO 300
        	    IF(ZEQ.AND.xtt1.EQ.CMZEQ)GOTO 300
        	    IF(ZGE.AND.xtt1.GE.CMZGE)GOTO 300
        	    IF(ZLE.AND.xtt2.LE.CMZLE)GOTO 300
        	    IF(ZEQ.AND.xtt2.EQ.CMZEQ)GOTO 300
        	    IF(ZGE.AND.xtt2.GE.CMZGE)GOTO 300
		    xt=sqrt(xtt1**2+xtt2**2)
		endif
* end of TMAP mod
        	IF(XT.LT.ZMIN)then
		    ZMIN=XT
		    jmn=j
		    imn=i
		endif
        	IF(XT.GT.ZMAX)then
		    ZMAX=XT
		    jmx=j
		    imx=i
		endif
300         CONTINUE
301         CONTINUE
	    if(abs(jtype).eq.2)then
		ivlen=nx*ny
		xhigh=x(ivlen+imx)
		xlow=x(ivlen+imn)
c
		ivlen=ivlen+nx
		yhigh=x(ivlen+jmx)
		ylow=x(ivlen+jmn)
	    else
		dx=(xmax-xmin)/MAX(nx-1,1)
		xhigh=xmin+dx*(imx-1)
		xlow=xmin+dx*(imn-1)
c
		dy=(ymax-ymin)/MAX(ny-1,1)
		yhigh=ymin+dy*(jmx-1)
		ylow=ymin+dy*(jmn-1)
	    endif
c
	    SYM='*PPL$ZMIN'
	    CALL PUTVAL(SYM,ZMIN,7,IER)
C
	    SYM='*PPL$ZMAX'
	    CALL PUTVAL(SYM,ZMAX,7,IER)
c
	    sym='*PPL$XHIGH'
	    call putval(sym,xhigh,7,IER)
c
	    sym='*PPL$YHIGH'
	    call putval(sym,yhigh,7,IER)
c
	    sym='*PPL$XLOW'
	    call putval(sym,xlow,7,IER)
c
	    sym='*PPL$YLOW'
	    call putval(sym,ylow,7,IER)

C       Don't define line symbols for FILLPOL/POLYGON
	ELSE if (makep .ne. 'FILLPOL' .AND. makep .ne. 'POLYGON') then
C
C	LINE DATA

	    WRITE(SYM,998)'XMIN',LINEN
998	    FORMAT('*PPL$',A,'(',I2,')')
	    CALL PUTVAL(SYM,XMIN1,7,IER)
C
	    WRITE(SYM,998)'XMAX',LINEN
	    CALL PUTVAL(SYM,XMAX1,7,IER)
C
	    WRITE(SYM,998)'YMIN',LINEN
	    CALL PUTVAL(SYM,YMIN1,7,IER)
C
	    WRITE(SYM,998)'YMAX',LINEN
	    CALL PUTVAL(SYM,YMAX1,7,IER)

* If its a ribbon plot, we need the min and max of the color-by variable.
	    IF (linen .EQ. 2) THEN
	       zmin = ymin1
	       zmax = ymax1
	    ENDIF

C
C	SET FIRST - LAST VALUES
C
	    IBASEX=0
	    IBASEY=NSIZE/2
	    DO 10 I=1,LINEN-1
	    LEN=LLENG(I)
	    IBASEX=IBASEX+LEN
	    IBASEY=IBASEY+LEN
10	    CONTINUE
	    LEN=LLENG(LINEN)
	    DO 20 J=1,LEN
	    IF((XLE.AND.X(IBASEX+J).LE.CMXLE).OR.
     *		(XEQ.AND.X(IBASEX+J).EQ.CMXEQ).OR.
     *		(XGE.AND.X(IBASEX+J).GE.CMXGE).OR.
     *		(YLE.AND.X(IBASEY+J).LE.CMYLE).OR.
     * 	(YEQ.AND.X(IBASEY+J).EQ.CMYEQ).OR.
     *		(YGE.AND.X(IBASEY+J).GE.CMYGE))GOTO 20
	    XT=X(IBASEX+J)*XFCT(I)+XOFF(I)
	    YT=X(IBASEY+J)*YFCT(I)+YOFF(I)
	    WRITE(SYM,998)'XFIRST',LINEN
	    CALL PUTVAL(SYM,XT,7,IER)
	    WRITE(SYM,998)'YFIRST',LINEN
	    CALL PUTVAL(SYM,YT,7,IER)
	    GOTO 30	
20	    CONTINUE
30	    DO 40 J=LEN,1,-1
	    IF((XLE.AND.X(IBASEX+J).LE.CMXLE).OR.
     *		(XEQ.AND.X(IBASEX+J).EQ.CMXEQ).OR.
     *		(XGE.AND.X(IBASEX+J).GE.CMXGE).OR.
     *		(YLE.AND.X(IBASEY+J).LE.CMYLE).OR.
     *		(YEQ.AND.X(IBASEY+J).EQ.CMYEQ).OR.
     *		(YGE.AND.X(IBASEY+J).GE.CMYGE))GOTO 40
	    XT=X(IBASEX+J)*XFCT(I)+XOFF(I)
	    YT=X(IBASEY+J)*YFCT(I)+YOFF(I)
	    WRITE(SYM,998)'XLAST',LINEN
	    CALL PUTVAL(SYM,XT,7,IER)
	    WRITE(SYM,998)'YLAST',LINEN
	    CALL PUTVAL(SYM,YT,7,IER)
	    GOTO 50
40	    CONTINUE
	ENDIF
50	RETURN
	END
