	SUBROUTINE TAYIS1(TLO,ITIME1,ITIME2,DT,NSMTIC,LABEL,IDDB,IMDB,
     *	TFACT,X,XFACT,CSIZE,xacross,hgrat,hcolor,hline)

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
*	Directory PMEL:[PLOT5.PPLV11.PLOTLIB]TAYIS1.FOR

* search for "TMAP" to find all revisions
* *sh*  - 1/23/89 added interrupt detection
*		  ... no explicit declarations (too much trouble !) so it must
*		      be compiled with /WARNINGS=NODECLARATIONS
C *acm* 1/24/00 Y2k Bug -- Put century at start  when comparing 
C                 ITIME to ITIME2 at loop 10
C *acm* 4/25/00 Further fix to the above - .LT. test rather than .LE.

C
* *jd* Mod for 4 digit years 11.27.95
C *kob* 04/07/96 passed in character strings had to be declared 
C                with "*(*)" for the linux port 
C *acm* 03/04  new arguments and code for plotting graticule lines across plot. 
C**
C**    @(#)tayis1.f	1.1    3/10/88
C**
C**
C***********************************************************************
C**
C**                 PLOT+ Scientific Graphics System
C**
C***********************************************************************
C**
C**
C
C     THIS ROUTINE PLOTS A TIME AXIS AND LABELS IT WITH THE DAY OF THE M
C     AND THE MONTH.
C     VARIBLES:
C         TLO=X VALUE CORRESPONDING TO THE START TIME.
C         ITIME1 = START TIME (IE.  7708190000)
C                  WE RESTRICT THE PERMISSIBLE START TIMES TO BE
C                  EVEN DAYS.  (THAT IS THE LAST 6 DIGITS OF ITIME1
C                  MUST BE 0)
C         ITIME2 = END TIME (IE   7709070000)
C                  THIS IS THE FIRST DAY NOT TO BE PLOTTED, AND IT
C                  MUST BE AN EVEN DAY.
C         DT = SAMPLING RATE IN MINUTES.
C         NSMTIC = NUMBER OF SMALL TICS TO MAKE EACH DAY.
C                = 23 FOR HOURLY TICS.
C                = 11 FOR BIHOURLY TICS.
C                   ETC.
C         LABEL > 0 IF LABELS ARE TO BE DRAWN ABOVE THE AXIS.
C               = 0 IF NO LABELS ARE TO BE DRAWN.
C               < 0 IF LABEL ARE TO BE DRAWN BELOW THE AXIS.
C         IABS(LABEL) = 1 FOR LABELS OF THE STYLE
C                  1..........1..........1..........1..........1
C                       3           4          5        6
C                                      JUN
C         IABS(LABEL) = 2 FOR LABELS OF THE STYLE
C                  1..........1..........1..........1..........1
C                     JUN  3     JUN  4      JUN  5    JUN  6
C         IDDB = PRINT LABELS OF DAYS DIVIBLE BY THIS NUMBER ONLY.
C                IF(IDDB = 0) PRINT NO DAY LABELS.
C         IMDB = PRINT LABELS OF MONTHS DIVISIBLE BY THIS NUMBER ONLY.
C                IF(IMDB = 0) PRINT NO MONTH LABELS.
C                THIS VARIBLE ONLY APPLYS WHEN IABS(LABEL) = 1
C         TFACT = (LENGTH OF T AXIS IN INCHES)/(LENGTH OF T AXIS IN USER
C         Y = Y COORDINATE OF TIME AXIS.
C         YFACT = (LENGTH OF Y AXIS IN INCHES)/(LENGTH OF Y AXIS IN USER
C         CSIZE = SIZE TO DRAW DAY LABELS.
C                 MONTH LABELS WILL BE 1.5 TIMES AS LARGE.
C
C     THERE WAS A MODIFICATION TO THIE IN JUNE 1980.  THE
C     SOLE PURPOSE FOR THIS MODIFICATION WAS TO IMPROVE 'S
C     EFFICIENCY.
C
	CHARACTER ITIME3*14
	CHARACTER SYM*21,ITIME1*(*),ITIME*14,ITIME2*(*),MON*3,DAY*2
	COMMON/TICS/TSMX,TLX,TSMY,TLY,IX,IY
	COMMON/AXLWID/XWID,YWID

* Parameters for drawing tics across plot

        REAL xacross
        REAL dash1_1, dash1_2, dash1_3, dash1_4
        REAL dash2_1, dash2_2, dash2_3, dash2_4
        REAL hgrat(4,2)        ! line type: 0=tics only, <0=line, >0=dash lengths 
        INTEGER hcolor(*)      ! pen number 
        LOGICAL hline          ! If true, do the lines across.
        LOGICAL solid1, solid2   ! solid lines across at tics
        LOGICAL dashed1, dashed2 ! dashed lines across at tics
        INTEGER ltic, color1, color2

* TMAP interrupt flag

	include 'xinterrupt.cmn'

C     CHECK THE PARAMETERS PASSED TO THE SUBROUTINE.
	IF(ITIME1(7:10).NE.'0000')STOP ',1'
	IF(ITIME2(7:10).NE.'0000')STOP ',2'

C set up for plotting graticules across at tic marks
        
        solid1 = (hgrat(1,1).LT.0 .AND. hgrat(2,1).LT.0 .AND.  
     .            hgrat(3,1).LT.0 .AND.  hgrat(4,1).LT.0 )
        solid2 = (hgrat(1,2).LT.0 .AND. hgrat(2,2).LT.0 .AND.  
     .            hgrat(3,2).LT.0 .AND.  hgrat(4,2).LT.0 )
        dashed1 = (hgrat(1,1).GT.0 .OR. hgrat(2,1).GT.0 .OR.  
     .            hgrat(3,1).GT.0 .OR.  hgrat(4,1).GT.0 )
        dashed2 = (hgrat(1,2).GT.0 .OR. hgrat(2,2).GT.0 .OR.  
     .            hgrat(3,2).GT.0 .OR.  hgrat(4,2).GT.0 )

        color1 = hcolor(1)
        color2 = hcolor(2)

        IF (dashed1) THEN
           dash1_1 = hgrat(1,1)
           dash1_2 = hgrat(2,1)
           dash1_3 = hgrat(3,1)
           dash1_4 = hgrat(4,1)
        ENDIF
        IF (dashed2) THEN
           dash2_1 = hgrat(1,2)
           dash2_2 = hgrat(2,2)
           dash2_3 = hgrat(3,2)
           dash2_4 = hgrat(4,2)
        ENDIF

C     SET PARAMETERS.

*     *jd Number of arguments of TIME3 calls reduced to 6.  TIME3 uses only
*     the first 6, 'seconds' are not handled.  Done for MAC 7.92

	TIC = 1440./DT
	NTIC = NSMTIC + 1
	IF(NSMTIC.LT.0)THEN
	    IHVY=-6
	    INML=-5
	ELSE
	    IHVY=-10
	    INML=-6
	    TDIST = TIC/NTIC
	ENDIF
	IF(LABEL+IY.EQ.0)THEN
	    XTEMP=0.0
	ELSE
	    XTEMP=TLY*0.5
	ENDIF
	IF(IDDB.NE.0)THEN
	    FIDDB=0.0
	ELSE
	    FIDDB=-1.4
	ENDIF
	IF(LABEL.LT.0) THEN
	    XTEMP1=X-(1.4*CSIZE+XTEMP)/XFACT
	    XTEMP2=X-((4.0+FIDDB)*CSIZE+XTEMP)/XFACT
	ELSE
* ... corrected tabulation error in next line *sh* 1/23/89
	    XTEMP1=X+(0.4*CSIZE+XTEMP)/XFACT
	    XTEMP2=X+((2.5+FIDDB)*CSIZE+XTEMP)/XFACT
	ENDIF
	IF(LABEL.NE.0)YWID=(4.0+FIDDB)*CSIZE+XTEMP


	T=TLO
	ITIME=ITIME1
	CALL TIME3(ITIME,IYR,IMON,IDAY,IHR,IMIN,ISEC,ICEN)
	TMON=T
	CALL PLOT(X,TLO,0,0)
	IF(IDAY.EQ.1)THEN
	    CALL PLOTT7(X,TLO,XFACT,TFACT,1,IHVY)
	ELSE
	    CALL PLOTT7(X,TLO,XFACT,TFACT,1,INML)
	ENDIF

c ----  count number of large tics. If no large tics will be drawn
c ----  then will want to draw graticule lines at the small ones.

      ITIME=ITIME1
      CALL TIME3(ITIME,IYR,IMON,IDAY,IHR,IMIN,ISEC,ICEN)

C      LOOP like loop 10 below
      ltic = 0
  110 CONTINUE
      IF ( interrupted ) RETURN		! TMAP interrupt
      
      CALL TIME1(1,ITIME,2,' ',I3,ITIME3,1440.)
      CALL TIME3(ITIME3,IYR3,IMON3,IDAY3,IHR3,IMIN3,ISEC3,ICEN3)

      ITIME=ITIME3
      IF(ITIME(13:14)//ITIME(1:12) .LT. 
     .       ITIME2(13:14)//ITIME2(1:12) ) THEN
         IF (IDAY3.EQ.1) ltic = ltic + 1  !It is a large tic.
         GO TO 110
      ENDIF

! IF no large tics, and if graticules at just the large tics were
! requested, then plot graticules at the small tics

	IF (ltic .EQ. 0) THEN   
           IF ( (.NOT.solid2) .AND. (solid1) ) THEN
              solid2 = solid1
              color2 = color1
           ENDIF
           IF ( (.NOT.dashed2) .AND. (dashed1) ) THEN 
              dashed2 = dashed1
              dash2_1 = dash1_1
              dash2_2 = dash1_2
              dash2_3 = dash1_3
              dash2_4 = dash1_4
              color2 = color1
           ENDIF
        ENDIF
c ----

      ITIME=ITIME1
      CALL TIME3(ITIME,IYR,IMON,IDAY,IHR,IMIN,ISEC,ICEN)

C      LOOP:
10	CALL PLOT(X,T,0,0)
	IF ( interrupted ) RETURN		! TMAP interrupt
	DO 20 I=1,NSMTIC
	CALL PLOTT7(X,T+FLOAT(I)*TDIST,XFACT,TFACT,1,-5)
        
                 IF ((solid2.OR.dashed2) .AND. hline) THEN ! make line across
                    CALL COLOR(color2)
                    IF (dashed2) THEN
                       CALL DASHSZ(dash2_1,dash2_2,dash2_3,dash2_4)
                       CALL DASHES
                    ENDIF
                    CALL PLOT(x,T+FLOAT(I)*TDIST,0,0) 
                    CALL PLOT(xacross,T+FLOAT(I)*TDIST,1,0)
                    CALL PLOT(x,T+FLOAT(I)*TDIST,0,0) 
                    CALL COLOR(1)
                    CALL VECTRS
                 ENDIF


20	CONTINUE
	CALL TIME1(1,ITIME,2,' ',I3,ITIME3,1440.)
	CALL TIME3(ITIME3,IYR3,IMON3,IDAY3,IHR3,IMIN3,ISEC3,ICEN3)
	IF(IDAY3.EQ.1)THEN
	    CALL PLOTT7(X,T+TIC,XFACT,TFACT,1,IHVY)
              IF ((solid1.OR.dashed1) .AND. hline) THEN ! make line across
                 CALL COLOR(color1)
                 IF (dashed1) THEN
                    CALL DASHSZ(dash1_1,dash1_2,dash1_3,dash1_4)
                    CALL DASHES
                 ENDIF
                 CALL PLOT(x,t+tic,0,0) 
                 CALL PLOT(xacross,t+tic,1,0)
                 CALL PLOT(x,t+tic,0,0) 
                 CALL COLOR(1)
                 CALL VECTRS
              ENDIF
	ELSE
	    CALL PLOTT7(X,T+TIC,XFACT,TFACT,1,INML)

                 IF ((solid2.OR.dashed2) .AND. hline) THEN ! make line across
                    CALL COLOR(color2)
                    IF (dashed2) THEN
                       CALL DASHSZ(dash2_1,dash2_2,dash2_3,dash2_4)
                       CALL DASHES
                    ENDIF
                    CALL PLOT(x,t+tic,0,0) 
                    CALL PLOT(xacross,t+tic,1,0)
                    CALL PLOT(x,t+tic,0,0) 
                    CALL COLOR(1)
                    CALL VECTRS
                 ENDIF

	ENDIF
C         THE LABELS
	CALL TIME3(ITIME,IYR,IMON,IDAY,IHR,IMIN,ISEC,ICEN)
	IF(IDDB.EQ.0.OR.LABEL.EQ.0) GO TO 50
	IF(IDAY.NE.IDDB*(IDAY/IDDB)) GO TO 50
	CALL TIME2(ITIME,SYM)
	READ(SYM,30) MON,DAY
30	FORMAT(6X,A3,1X,A2)
	IF(DAY(1:1).EQ.' ')THEN
	    NC=1
	    DAY=DAY(2:2)
	ELSE
	    NC=2
	ENDIF
	IF(IABS(LABEL).EQ.1)THEN
	    TSYM=T+(TIC-SYMWID(CSIZE,NC,DAY)/TFACT)*.5
	    CALL SYMBEL(XTEMP1,TSYM,90.,CSIZE,NC,DAY)
	ELSE
	    WRITE(SYM,40)MON,DAY
40	    FORMAT(A3,' ',A2)
	    TSYM=T+(TIC-SYMWID(CSIZE,6,SYM)/TFACT)*.5
	    CALL SYMBEL(XTEMP1,TSYM,90.,CSIZE,6,SYM)
	ENDIF
50	CONTINUE
	IF(IMDB.EQ.0.OR.IABS(LABEL).NE.1) GO TO 70
	IF(IMON.EQ.IMON3) GO TO 70
	IF(IMON.NE.IMDB*(IMON/IMDB)) GO TO 60
C             ELSE:
	IF(T-TMON.GT.4320./DT)THEN
	    CALL TIME2(ITIME,SYM)
	    READ(SYM,30)MON,DAY
	    TSYM=TMON+(T+TIC-TMON-SYMWID(1.5*CSIZE,3,MON)/TFACT)*.5
	    CALL SYMBEL(XTEMP2,TSYM,90.,1.5*CSIZE,3,MON)
	ENDIF
60	TMON=T+TIC
70	CONTINUE
	ITIME=ITIME3
	T=T+TIC
c	IF(ITIME.LT.ITIME2) GO TO 10
        IF(ITIME(13:14)//ITIME(1:12) .LT.
     .       ITIME2(13:14)//ITIME2(1:12)) GO TO 10
     
	IF(IMDB.EQ.0) RETURN
	IF(IABS(LABEL).EQ.1.AND.IMON.EQ.IMDB*(IMON/IMDB)
     .	.AND.IDAY3.GE.4)THEN
	    CALL TIME2(ITIME,SYM)
	    READ(SYM,30)MON,DAY
	    TSYM=TMON+(T-TMON-SYMWID(1.5*CSIZE,3,MON)/TFACT)*.5
	    CALL SYMBEL(XTEMP2,TSYM,90.,1.5*CSIZE,3,MON)
	ENDIF
	RETURN
	END
