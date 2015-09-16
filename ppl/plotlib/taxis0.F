      SUBROUTINE TAXIS0(TLO,ITIME1,ITIME2,DT,NSMTIC,LABEL,IHDB,IDDB,
     1    TFACT,Y,YFACT,CSIZE,yacross,vgrat,vcolor,vline,color0,is_hours_axis)

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
*	Directory PMEL:[PLOT5.PPLV11.PLOTLIB]TAXIS0.FOR

* search for "TMAP" to find all revisions
* *sh*  - 1/23/89 added interrupt detection
*		  ... no explicit declarations (too much trouble !) so it must
*		      be compiled with /WARNINGS=NODECLARATIONS
* *sh*  - 4/19/89 FERRET 1.21  (repeat of correction made earlier)
*       - changed "mon" to "date" in calls to SYMWID and SYMBEL.
*         "mon" is the wrong data type                          
                                                                
* *jd* Mod for 4 digit years 11.27.95
C *kob* 04/07/96 passed in character strings had to be declared 
C                with "*(*)" for the linux port 
C *acm* 24/01/00 Y2k Bug -- Put century at start when comparing 
C                 ITIME to ITIME2 at loop 10
C *acm* 03/04  new arguments and code for plotting graticule lines across plot. 
C V65  *acm*  1/10 New time-axis style for DAY and MONTH, DMY: adds the
C                  year to DAY,MONTH or to MONTH.
C V695  *acm*  9/15 Changes for hour:min labeling on time axes

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
	CHARACTER ITIME3*14,DATE*6,BUFF*14,YEAR*4
	CHARACTER ITIME*14,ITIME1*14,ITIME2*14,SYM*21
	COMMON/TICS/TSMX,TLX,TSMY,TLY,IX,IY
	COMMON/AXLWID/XWID,YWID

	REAL*8  DIFF, difft
	REAL*8  td, ytemp1, tfact_d, yfact_d
	REAL    delt, rem
	INTEGER is_hours_axis, start_hr

* Parameters for drawing tics across plot

        REAL yacross
        REAL dash1_1, dash1_2, dash1_3, dash1_4
        REAL dash2_1, dash2_2, dash2_3, dash2_4
        REAL vgrat(4,2)        ! line type: 0=tics only, <0=line, >0=dash lengths 
        INTEGER vcolor(*)      ! pen number 
        LOGICAL vline          ! If true, do the lines across.
        LOGICAL solid1, solid2   ! solid lines across at tics
        LOGICAL dashed1, dashed2 ! dashed lines across at tics
        INTEGER ltic, color1, color2

* TMAP interrupt flag
	include 'xinterrupt.cmn'
        include 'century.cmn'  ! with MONYRLAB flag

	
  80	FORMAT(I2.2)
  82	FORMAT(I2.2,':00')
  84	FORMAT(I2.2,':',I2.2)

	difft = DIFF(ITIME1,ITIME2)

C     CHECK THE PARAMETERS PASSED TO THE SUBROUTINE.

	IF (is_hours_axis .EQ. 0) THEN
	   IF(ITIME1(9:10).NE.'00')STOP ',1'  ! MINUTE,SECOND ARE 0
	   IF(ITIME2(9:10).NE.'00')STOP ',2'
	ENDIF

        IF(ITIME1(13:14)//ITIME1(1:12).GE.
     .	   ITIME2(13:14)//ITIME2(1:12)) STOP',3'

C set up for plotting graticules across at tic marks
        
        solid1 = (vgrat(1,1).LT.0 .AND. vgrat(2,1).LT.0 .AND.  
     .            vgrat(3,1).LT.0 .AND.  vgrat(4,1).LT.0 )
        solid2 = (vgrat(1,2).LT.0 .AND. vgrat(2,2).LT.0 .AND.  
     .            vgrat(3,2).LT.0 .AND.  vgrat(4,2).LT.0 )
        dashed1 = (vgrat(1,1).GT.0 .OR. vgrat(2,1).GT.0 .OR.  
     .            vgrat(3,1).GT.0 .OR.  vgrat(4,1).GT.0 )
        dashed2 = (vgrat(1,2).GT.0 .OR. vgrat(2,2).GT.0 .OR.  
     .            vgrat(3,2).GT.0 .OR.  vgrat(4,2).GT.0 )

        color1 = vcolor(1)
        color2 = vcolor(2)

        IF (dashed1) THEN
           dash1_1 = vgrat(1,1)
           dash1_2 = vgrat(2,1)
           dash1_3 = vgrat(3,1)
           dash1_4 = vgrat(4,1)
        ENDIF
        IF (dashed2) THEN
           dash2_1 = vgrat(1,2)
           dash2_2 = vgrat(2,2)
           dash2_3 = vgrat(3,2)
           dash2_4 = vgrat(4,2)
        ENDIF
C
C     SET PARAMETERS.
        TIC = 60./DT  
        TICd = 60./DT  
        NTIC = NSMTIC + 1
        delt = 60.
	labint = 120

	IF (is_hours_axis.EQ.1) THEN
	   delt = 10.
	   IF (difft .LT. 50) delt = 5.
	   IF (difft .LT. 30) delt = 2.
	   IF (difft .LT. 12) delt = 1.
	   tic = delt/dt
	   ticd = delt/dt
	ENDIF
      
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
	YTEMP1=Y+DBLE(LABEL*(0.9*CSIZE+YTEMP)/YFACT)
      IF(LABEL.LT.0) THEN
      YTEMP2=Y-((4.0+FIHDB)*CSIZE+YTEMP)/YFACT
      ELSE
      YTEMP2=Y+((2.5+FIHDB)*CSIZE+YTEMP)/YFACT
      ENDIF
	IF(LABEL.NE.0)XWID=(4.0+FIHDB)*CSIZE+YTEMP
      T=TLO  ! Time in hours, plot units
      td = tlo
      ITIME=ITIME1
      CALL TIME3(ITIME,IYR,IMON,IDAY,IHR,IMIN,ISEC,ICEN)
      TDAY=T

      tfact_d = tfact
      yfact_d = yfact
	
      CALL PLOT(TLO,Y,0,0)
      IF(IHR.EQ.0)THEN
	CALL PLOTT7(TLO,Y,TFACT,YFACT,1,IHVY)  ! draw to the location and make a mark
      ELSE
	CALL PLOTT7(TLO,Y,TFACT,YFACT,1,INML)
      ENDIF
	
      CALL TIME1(1,ITIME,2,' ',I3,ITIME3,delt)  ! next time stepping from 1 (arg 1) to 2 (arg 2) -- return ITIME3
      ITIME = ITIME3

c Check whether to make an hour label at the start of the axis
c ihdb is hour-label frequency
      IF(IABS(LABEL).EQ.1) THEN
         IF (IHR.EQ.IHDB*(IHR/IHDB)) THEN 
            WRITE(SYM,80) IHR
	    CALL scinot_d(Td,YTEMP1,0.,CSIZE,2,SYM,0,wid,tfact_d, yfact_d)
	 ENDIF
         IF (is_hours_axis .EQ. 1) THEN
	    IF (IMIN .EQ. 0) THEN
	       start_hr = -1  ! marks whether hour is labeled at the start of the axis
               WRITE(SYM,82) IHR
	       CALL scinot_d(Td,YTEMP1,0.,CSIZE,5,SYM,0,wid,tfact_d, yfact_d)
	    ELSE ! get the loop on-track to hit a tic mark at the start of hours
	       rem = delt*INT(imin/delt) - imin
	       ITIME=ITIME1
	       CALL TIME3(ITIME,IYR,IMON,IDAY,IHR,IMIN,ISEC,ICEN)
	       CALL TIME1(1,ITIME,2,' ',I3,ITIME3,rem) 
	       ITIME1 = ITIME3
	    ENDIF

	 ENDIF
         IF (is_hours_axis.EQ.1 .AND. IMIN.NE.0) start_hr = IHR  ! need to label hour on 1st Tic
      ENDIF

c ----  count number of large tics. If no large tics will be drawn
c ----  then will want to draw graticule lines at the small ones.

c Set itime1, itime3 as above

      ITIME=ITIME1
      CALL TIME3(ITIME,IYR,IMON,IDAY,IHR,IMIN,ISEC,ICEN)
      CALL TIME1(1,ITIME,2,' ',I3,ITIME3,delt) 
      ITIME = ITIME3

C      LOOP:
       labint = 120
       ltic = 0
  110  CONTINUE

	CALL TIME3(ITIME,IYR,IMON,IDAY,IHR,IMIN,ISEC,ICEN)
  	CALL TIME1(1,ITIME,2,' ',I3,ITIME3,delt)
	ITIME=ITIME3

	IF(ITIME(13:14)//ITIME(1:12) .LE.
     .       ITIME2(13:14)//ITIME2(1:12)) THEN
           IF(IHR.EQ.0) ltic = ltic + 1
	   IF (is_hours_axis.EQ.1 .AND. imin.EQ.0. AND. ihr.NE.0) ltic = ltic+1
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
	
! On short-time interval axes, label interval
	IF (ltic.LT.2 .AND. is_hours_axis.GT.0) labint = 2*delt

! Restore the state of ITIME, ITIME3
      ITIME=ITIME1
      CALL TIME3(ITIME,IYR,IMON,IDAY,IHR,IMIN,ISEC,ICEN)

      CALL TIME1(1,ITIME,2,' ',I3,ITIME3,delt)
      ITIME = ITIME3

c ----

C      LOOP:
   10     CALL PLOT(T,Y,0,0)
	  IF ( interrupted ) RETURN		! TMAP interrupt
          DO 20 I=1,NSMTIC
              CALL PLOTT7(T+FLOAT(I)*TDIST,Y,TFACT,YFACT,1,-7)
     
                 IF ((solid2.OR.dashed2) .AND. vline) THEN ! make line across
                    CALL COLOR(color2)
                    IF (dashed2) THEN
                       CALL DASHSZ(dash2_1,dash2_2,dash2_3,dash2_4)
                       CALL DASHES
                    ENDIF
                    CALL PLOT(T+FLOAT(I)*TDIST,y,0,0) 
                    CALL PLOT(T+FLOAT(I)*TDIST,yacross,1,0)
                    CALL PLOT(T+FLOAT(I)*TDIST,Y,0,0) 
                     CALL COLOR(color0)
                    CALL VECTRS
                 ENDIF

   20         CONTINUE
          CALL TIME3(ITIME,IYR,IMON,IDAY,IHR,IMIN,ISEC,ICEN)
	IF(IHR.EQ.0)THEN
	CALL PLOTT7(T+TIC,Y,TFACT,YFACT,1,IHVY)
              IF ((solid1.OR.dashed1) .AND. vline) THEN ! make line across
                 CALL COLOR(color1)
                 IF (dashed1) THEN
                    CALL DASHSZ(dash1_1,dash1_2,dash1_3,dash1_4)
                    CALL DASHES
                 ENDIF
                 CALL PLOT(t+tic,y,0,0) 
                 CALL PLOT(t+tic,yacross,1,0)
                 CALL PLOT(t+tic,Y,0,0) 
                  CALL COLOR(color0)
                 CALL VECTRS
              ENDIF
      
	ELSE
	       CALL PLOTT7(T+TIC,Y,TFACT,YFACT,1,INML)
     
               IF ((solid1.OR.dashed1) .AND. vline) THEN ! make line across
                  CALL COLOR(color1)
                  IF (dashed1) THEN
                     CALL DASHSZ(dash1_1,dash2_1,dash1_3,dash1_4)
                     CALL DASHES
                  ENDIF
                  CALL PLOT(t+tic,y,0,0) 
                  CALL PLOT(t+tic,yacross,1,0)
                  CALL PLOT(t+tic,Y,0,0) 
                   CALL COLOR(color0)
                  CALL VECTRS
               ENDIF

	ENDIF
C         THE LABELS

	  IF (IABS(LABEL).EQ.1) THEN
             IF (is_hours_axis.EQ.1) THEN
	        IF (imin .EQ. 0) THEN
	           WRITE(SYM,82) IHR
	           CALL scinot_d(Td+TICd,YTEMP1,0.,CSIZE,5,SYM,0,wid,tfact_d, yfact_d)
		ELSE
                   IF (MOD(imin,labint) .EQ. 0) THEN
		      IF (start_hr .GT. 0) THEN
	                 WRITE(SYM,84) start_hr, imin
		         start_hr = -1
		      ELSE
	                 WRITE(SYM,84) ihr, imin
		      ENDIF
	              CALL scinot_d(Td+TICd,YTEMP1,0.,CSIZE,5,SYM,0,wid,tfact_d, yfact_d)
		   ENDIF
		ENDIF
             ELSE
	        IF(IHR.EQ.IHDB*(IHR/IHDB)) THEN
	           WRITE(SYM,80) IHR
	           CALL scinot_d(Td+TICd,YTEMP1,0.,CSIZE,2,SYM,0,wid,tfact_d, yfact_d)
	        ENDIF
             ENDIF

	  ENDIF

          CALL TIME1(1,ITIME,2,' ',I3,ITIME3,delt)
          IF(IDDB.EQ.0.OR.IABS(LABEL).NE.1) GO TO 70
              CALL TIME3(ITIME3,IYR3,IMON3,IDAY3,IHR3,IMIN3,ISEC3,ICEN3)
              IF(IDAY.EQ.IDAY3) GO TO 70
              IF(IDAY.NE.IDDB*(IDAY/IDDB))GO TO 60
C             ELSE:
	IF(T-TDAY.GT.180./DT)THEN
           CALL TIME2(ITIME,SYM)
	   READ(SYM,55) DATE, YEAR
55	   FORMAT(6X,A6,1X,A4)
65         FORMAT(A6,1X,A4)
           IF (MONYRLAB) THEN
	      WRITE (BUFF, 65) DATE, YEAR
              TSYM=TDAY+(T+TIC-TDAY-SYMWID(CSIZE,14,BUFF)/TFACT)*.5
              CALL SYMBEL(TSYM,YTEMP2,0.,CSIZE,14,BUFF)
	   ELSE
              TSYM=TDAY+(T+TIC-TDAY-SYMWID(1.5*CSIZE,6,DATE)/TFACT)*.5
	      CALL SYMBEL(TSYM,YTEMP2,0.,1.5*CSIZE,6,DATE)
	   ENDIF
	ENDIF
   60   TDAY=T+TIC
   70     CONTINUE
          ITIME=ITIME3
          T=T+TIC
	  td = td + ticd
c          IF(ITIME.LE.ITIME2) GO TO 10

          IF(ITIME(13:14)//ITIME(1:12) .LE.
     .       ITIME2(13:14)//ITIME2(1:12)) GO TO 10

      IF(IDDB.EQ.0) RETURN

      IF(IABS(LABEL).EQ.1.AND.IDAY.EQ.IDDB*(IDAY/IDDB) ) THEN
         IF (is_hours_axis.EQ.1 .OR. IHR3.GE.4) THEN
           CALL TIME2(ITIME,SYM)
	   READ(SYM,55) DATE, YEAR
	   IF (MONYRLAB) THEN
	      WRITE (BUFF, 65) DATE, YEAR
              TSYM=TDAY+(T-TDAY-SYMWID(CSIZE,14,BUFF)/TFACT)*.5
              CALL SYMBEL(TSYM,YTEMP2,0.,CSIZE,14,BUFF)
           ELSE
              TSYM=TDAY+(T-TDAY-SYMWID(1.5*CSIZE,6,DATE)/TFACT)*.5
	      CALL SYMBEL(TSYM,YTEMP2,0.,1.5*CSIZE,6,DATE)
           ENDIF

         ENDIF
      ENDIF

      RETURN
      END
