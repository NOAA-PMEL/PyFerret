	SUBROUTINE TAYIS0(TLO,ITIME1,ITIME2,DT,NSMTIC,LABEL,IHDB,IDDB,
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
*	Directory PMEL:[PLOT5.PPLV11.PLOTLIB]TAYIS0.FOR

* search for "TMAP" to find all revisions
* *sh*  - 1/23/89 added interrupt detection
*		  ... no explicit declarations (too much trouble !) so it must
*		      be compiled with /WARNINGS=NODECLARATIONS
* *sh*  - 8/11/89 FERRET 2.00 (modelled after TAXIS0 correction)
*       - changed "mon" to "date" in calls to SYMWID and SYMBEL.
*         "mon" is the wrong data type                          

* *jd* Mod for 4 digit years 11.27.95
C *kob* 04/07/96 passed in character strings had to be declared 
C                with "*(*)" for the linux port 
C *acm* 24/01/00 Y2k Bug --Put century at start when comparing 
C                 ITIME to ITIME2 at loop 10
C *acm* 03/04  new arguments and code for plotting graticule lines across plot. 

C**
C**    @(#)tayis0.f	1.1    3/10/88
C**
C**
C***********************************************************************
C**
C**                 PLOT+ Scientific Graphics System
C**
C***********************************************************************
C**
C**
	CHARACTER ITIME3*14,DATE*6
	CHARACTER ITIME*14,ITIME1*(*),ITIME2*(*),SYM*21
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

* TMAP interrupt flag

	include 'xinterrupt.cmn'

C     THIS ROUTINE PLOTS A TIME AXIS AND LABELS IT WITH THE DAY OF THE M
C     AND THE MONTH.
C     VARIBLES:
C         TLO=X VALUE CORRESPONDING TO THE START TIME.
C         ITIME1 = START TIME (IE.  7708190100)
C                  WE RESTRICT THE PERMISSIBLE START TIMES TO BE
C                  EVEN HOURS.  (THAT IS THE LAST 4 DIGITS OF ITIME1
C                  MUST BE 0)
C         ITIME2 = END TIME (IE   7709072300)
C                  THIS IS THE FIRST HOUR NOT TO BE PLOTTED, AND IT
C                  MUST BE AN EVEN HOUR.
C         DT = SAMPLING RATE IN MINUTES.
C         NSMTIC = NUMBER OF SMALL TICS TO MAKE EACH HOUR.
C         LABEL > 0 IF LABELS ARE TO BE DRAWN ABOVE THE AXIS.
C               = 0 IF NO LABELS ARE TO BE DRAWN.
C               < 0 IF LABEL ARE TO BE DRAWN BELOW THE AXIS.
C         IABS(LABEL) = 1 FOR LABELS OF THE STYLE
C                  1..........1..........1..........1..........1
C                       3           4          5        6
C                                    JUN 7
C         IHDB = PRINT LABELS OF HOURS DIVIBLE BY THIS NUMBER ONLY.
C                IF(IHDB = 0) PRINT NO DAY LABELS.
C         IDDB = PRINT LABELS OF DAYS DIVISIBLE BY THIS NUMBER ONLY.
C                IF(IDDB = 0) PRINT NO MONTH LABELS.
C                THIS VARIBLE ONLY APPLYS WHEN IABS(LABEL) = 1
C         TFACT = (LENGTH OF T AXIS IN INCHES)/(LENGTH OF T AXIS IN USER
C         Y = Y COORDINATE OF TIME AXIS.
C         YFACT = (LENGTH OF Y AXIS IN INCHES)/(LENGTH OF Y AXIS IN USER
C         CSIZE = SIZE TO DRAW HOUR LABELS.
C                 DAY LABELS WILL BE 1.5 TIMES AS LARGE.
C
C
C     THIS ROUTINE WAS FIRST WRITTEN BY DAVID MANDEL IN JULY 1980.
C
C     CHECK THE PARAMETERS PASSED TO THE SUBROUTINE.
	IF(ITIME1(9:10).NE.'00')STOP ',1'
	IF(ITIME2(9:10).NE.'00')STOP ',2'
        IF(ITIME1(13:14)//ITIME1(1:12).GE.
     .	   ITIME2(13:14)//ITIME2(1:12)) STOP',3'
        
        solid1 = (hgrat(1,1).LT.0 .AND. hgrat(2,1).LT.0 .AND.  
     .            hgrat(3,1).LT.0 .AND.  hgrat(4,1).LT.0 )
        solid2 = (hgrat(1,2).LT.0 .AND. hgrat(2,2).LT.0 .AND.  
     .            hgrat(3,2).LT.0 .AND.  hgrat(4,2).LT.0 )
        dashed1 = (hgrat(1,1).GT.0 .OR. hgrat(2,1).GT.0 .OR.  
     .            hgrat(3,1).GT.0 .OR.  hgrat(4,1).GT.0 )
        dashed2 = (hgrat(1,2).GT.0 .OR. hgrat(2,2).GT.0 .OR.  
     .            hgrat(3,2).GT.0 .OR.  hgrat(4,2).GT.0 )

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
C
C     SET PARAMETERS.
	TIC = 60./DT
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
	IF(IHDB.NE.0)THEN
	    FIHDB=0.0
	ELSE
	    FIHDB=-1.4
	ENDIF
	XTEMP1=X+LABEL*(0.9*CSIZE+XTEMP)/XFACT
	IF(LABEL.LT.0) THEN
	    XTEMP2=X-((4.0+FIHDB)*CSIZE+XTEMP)/XFACT
	ELSE
	    XTEMP2=X+((2.5+FIHDB)*CSIZE+XTEMP)/XFACT
	ENDIF
	IF(LABEL.NE.0)YWID=(4.0+FIHDB)*CSIZE+XTEMP
	T=TLO
	ITIME=ITIME1
	CALL TIME3(ITIME,IYR,IMON,IDAY,IHR,IMIN,ISEC,ICEN)
	TDAY=T
	CALL PLOT(X,TLO,0,0)
	IF(IHR.EQ.0)THEN
	    CALL PLOTT7(X,TLO,XFACT,TFACT,1,IHVY)
	ELSE
	    CALL PLOTT7(X,TLO,XFACT,TFACT,1,INML)
	ENDIF
	CALL TIME1(1,ITIME,2,' ',I3,ITIME3,60.)
	ITIME = ITIME3
	IF(IABS(LABEL).NE.1.OR.IHR.NE.IHDB*(IHR/IHDB)) GO TO 10
C      ELSE:
	WRITE(SYM,80) IHR
80	FORMAT(I2.2)
	CALL SCINOT(XTEMP1,T,90.,CSIZE,2,SYM,0,WID,XFACT,TFACT)
C      LOOP:
10	CALL PLOT(X,T,0,0)
	IF ( interrupted ) RETURN		! TMAP interrupt
	DO 20 I=1,NSMTIC
	CALL PLOTT7(X,T+FLOAT(I)*TDIST,XFACT,TFACT,1,-5)
        
                 IF ((solid2.OR.dashed2) .AND. hline) THEN ! make line across
                    CALL COLOR(hcolor(2))
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
	CALL TIME3(ITIME,IYR,IMON,IDAY,IHR,IMIN,ISEC,ICEN)
	IF(IHR.EQ.0)THEN
	    CALL PLOTT7(X,T+TIC,XFACT,TFACT,1,IHVY)
            
              IF ((solid1.OR.dashed1) .AND. hline) THEN ! make line across
                 CALL COLOR(hcolor(1))
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
            
                 IF ((solid1.OR.dashed1) .AND. hline) THEN ! make line across
                    CALL COLOR(hcolor(1))
                    IF (dashed1) THEN
                       CALL DASHSZ(dash1_1,dash2_1,dash1_3,dash1_4)
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
	IF(IHDB.EQ.0.OR.IABS(LABEL).NE.1) GO TO 50
	IF(IHR.NE.IHDB*(IHR/IHDB)) GO TO 50
	WRITE(SYM,80) IHR
	CALL SCINOT(XTEMP1,T+TIC,90.,CSIZE,2,SYM,0,WID,XFACT,TFACT)
50	CONTINUE
	CALL TIME1(1,ITIME,2,' ',I3,ITIME3,60.)
	IF(IDDB.EQ.0.OR.IABS(LABEL).NE.1) GO TO 70
	CALL TIME3(ITIME3,IYR3,IMON3,IDAY3,IHR3,IMIN3,ISEC3,ICEN3)
	IF(IDAY.EQ.IDAY3) GO TO 70
	IF(IDAY.NE.IDDB*(IDAY/IDDB))GO TO 60
C             ELSE:
	IF(T-TDAY.GT.180./DT)THEN
	    CALL TIME2(ITIME,SYM)
	    READ(SYM,55) DATE
55	    FORMAT(6X,A6)
	    TSYM=TDAY+(T+TIC-TDAY-SYMWID(1.5*CSIZE,6,DATE)/TFACT)*.5
	    CALL SYMBEL(XTEMP2,TSYM,90.,1.5*CSIZE,6,DATE)
	ENDIF
60	TDAY=T+TIC
70	CONTINUE
	ITIME=ITIME3
	T=T+TIC
c	IF(ITIME.LE.ITIME2) GO TO 10
        IF(ITIME(13:14)//ITIME(1:12) .LE.
     .       ITIME2(13:14)//ITIME2(1:12)) GO TO 10
	IF(IDDB.EQ.0) RETURN
	IF(IABS(LABEL).EQ.1.AND.IDAY.EQ.IDDB*(IDAY/IDDB)
     .	.AND.IHR3.GE.4)THEN
	    CALL TIME2(ITIME,SYM)
	    READ(SYM,55) DATE
* TMAP mod 8/11/89 *sh*                                 
! original:  ("mon" --> "date")                           
!	    TSYM=TDAY+(T-TDAY-SYMWID(1.5*CSIZE,6,MON)/TFACT)*.5
!	    CALL SYMBEL(XTEMP2,TSYM,90.,1.5*CSIZE,6,MON)
	    TSYM=TDAY+(T-TDAY-SYMWID(1.5*CSIZE,6,date)/TFACT)*.5
	    CALL SYMBEL(XTEMP2,TSYM,90.,1.5*CSIZE,6,date)
	ENDIF
	RETURN
	END
