      SUBROUTINE TAXIS0(TLO,ITIME1,ITIME2,DT,NSMTIC,LABEL,IHDB,IDDB,
     1    TFACT,Y,YFACT,CSIZE)

* revised for TMAP
* from PPLUS :
*	Directory PMEL:[PLOT5.PPLV11.PLOTLIB]TAXIS0.FOR

* search for "TMAP" to find all revisions
* *sh*  - 1/23/89 added interrupt detection
*		  ... no explicit declarations (too much trouble !) so it must
*		      be compiled with /WARNINGS=NODECLARATIONS
* *sh*  - 4/19/89 FERRET 1.21  (repeat of correction made earlier)
*       - changed "mon" to "date" in calls to SYMWID and SYMBEL.
*         "mon" is the wrong data type                          
                                                                
* *jd* Mod for 4 digit years 11.27.95
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
C                with "*(*)" for the linux port 

C**
C**    @(#)taxis0.f	1.1    3/10/88
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

* TMAP interrupt flag
#ifdef unix
	include 'tmap_pplv11inc/xinterrupt.cmn'
#else
	INCLUDE 'tmap_pplv11inc:tmap_xinterrupt.cmn'
#endif

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
C
C     SET PARAMETERS.
      TIC = 60./DT
      NTIC = NSMTIC + 1
	IF(NSMTIC.LT.0)THEN
		IHVY=-8
		INML=-7
	ELSE
		IHVY=-9
		INML=-8
		TDIST = TIC/NTIC
	ENDIF
	IF(LABEL+IX.EQ.0)THEN
		YTEMP=0.0
	ELSE
		YTEMP=TLX*0.5
	ENDIF
	IF(IHDB.NE.0)THEN
	FIHDB=0.0
	ELSE
	FIHDB=-1.4
	ENDIF
	YTEMP1=Y+LABEL*(0.9*CSIZE+YTEMP)/YFACT
      IF(LABEL.LT.0) THEN
      YTEMP2=Y-((4.0+FIHDB)*CSIZE+YTEMP)/YFACT
      ELSE
      YTEMP2=Y+((2.5+FIHDB)*CSIZE+YTEMP)/YFACT
      ENDIF
	IF(LABEL.NE.0)XWID=(4.0+FIHDB)*CSIZE+YTEMP
      T=TLO
      ITIME=ITIME1
      CALL TIME3(ITIME,IYR,IMON,IDAY,IHR,IMIN,ISEC,ICEN)
      TDAY=T
      CALL PLOT(TLO,Y,0,0)
	IF(IHR.EQ.0)THEN
	CALL PLOTT7(TLO,Y,TFACT,YFACT,1,IHVY)
	ELSE
	CALL PLOTT7(TLO,Y,TFACT,YFACT,1,INML)
	ENDIF
      CALL TIME1(1,ITIME,2,' ',I3,ITIME3,60.)
      ITIME = ITIME3
      IF(IABS(LABEL).NE.1.OR.IHR.NE.IHDB*(IHR/IHDB)) GO TO 10
C      ELSE:
          WRITE(SYM,80) IHR
80	FORMAT(I2.2)
	CALL SCINOT(T,YTEMP1,0.,CSIZE,2,SYM,0,WID,TFACT,YFACT)
C      LOOP:
   10     CALL PLOT(T,Y,0,0)
	  IF ( interrupted ) RETURN		! TMAP interrupt
          DO 20 I=1,NSMTIC
              CALL PLOTT7(T+FLOAT(I)*TDIST,Y,TFACT,YFACT,1,-7)
   20         CONTINUE
          CALL TIME3(ITIME,IYR,IMON,IDAY,IHR,IMIN,ISEC,ICEN)
	IF(IHR.EQ.0)THEN
	CALL PLOTT7(T+TIC,Y,TFACT,YFACT,1,IHVY)
	ELSE
	       CALL PLOTT7(T+TIC,Y,TFACT,YFACT,1,INML)
	ENDIF
C         THE LABELS
          IF(IHDB.EQ.0.OR.IABS(LABEL).NE.1) GO TO 50
          IF(IHR.NE.IHDB*(IHR/IHDB)) GO TO 50
	WRITE(SYM,80) IHR
	CALL SCINOT(T+TIC,YTEMP1,0.,CSIZE,2,SYM,0,WID,TFACT,YFACT)
   50     CONTINUE
          CALL TIME1(1,ITIME,2,' ',I3,ITIME3,60.)
          IF(IDDB.EQ.0.OR.IABS(LABEL).NE.1) GO TO 70
              CALL TIME3(ITIME3,IYR3,IMON3,IDAY3,IHR3,IMIN3,ISEC3,ICEN3)
              IF(IDAY.EQ.IDAY3) GO TO 70
              IF(IDAY.NE.IDDB*(IDAY/IDDB))GO TO 60
C             ELSE:
	IF(T-TDAY.GT.180./DT)THEN
        CALL TIME2(ITIME,SYM)
	READ(SYM,55) DATE
55	FORMAT(6X,A6)
        TSYM=TDAY+(T+TIC-TDAY-SYMWID(1.5*CSIZE,6,DATE)/TFACT)*.5
        CALL SYMBEL(TSYM,YTEMP2,0.,1.5*CSIZE,6,DATE)
	ENDIF
   60         TDAY=T+TIC
   70     CONTINUE
          ITIME=ITIME3
          T=T+TIC
          IF(ITIME.LE.ITIME2) GO TO 10
      IF(IDDB.EQ.0) RETURN
	IF(IABS(LABEL).EQ.1.AND.IDAY.EQ.IDDB*(IDAY/IDDB)
     .	.AND.IHR3.GE.4)THEN
      CALL TIME2(ITIME,SYM)
	READ(SYM,55) DATE
* TMAP mod 9/17/89 *sh*                                 
! original:  ("mon" --> "date")                           
!      TSYM=TDAY+(T-TDAY-SYMWID(1.5*CSIZE,6,MON)/TFACT)*.5
!      CALL SYMBEL(TSYM,YTEMP2,0.,1.5*CSIZE,6,MON)
	TSYM=TDAY+(T-TDAY-SYMWID(1.5*CSIZE,6,date)/TFACT)*.5
	CALL SYMBEL(TSYM,YTEMP2,0.,1.5*CSIZE,6,date)        
* end of 9/17/89 TMAP mod
      ENDIF
      RETURN
      END
