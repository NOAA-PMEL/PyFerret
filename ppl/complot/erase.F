      SUBROUTINE ERASE
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
C**    @(#)erase.F	1.2    3/18/88
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
C     CALL ERASE
C
C     THIS SUBROUTINE IS USED TO INDICATE THAT A DISPLAY HAS
C     BEEN COMPLETED AND A NEW DISPLAY IS ABOUT TO BEGIN.  ON
C     THE TEKTERMINAL, THE SCREEN IS ERASED AND THE ALPHA CUR-
C     SOR IS MOVED TO THE UPPER LEFT-HAND CORNER OF THE SCREEN.
C     ON THE PRINTRONIX PRINTER PLOTTER, THE RASTER BUFFER IS
C     RE-INITIALIZED.  ON THE ZETA AND BENSON
C     PLOTTERS, IF PLOTTING HAS BEEN DONE SINCE THE LAST CALL
C     TO PLOTEND OR ERASE, THE PEN IS LIFTED AND MOVED TO THE
C     LEFT MARGIN THREE INCHES ABOVE THE TOP OF THE LAST PLOT.
C     IN ALL CASES, THE MODE IS SET TO ALPHAS.
C
C
C
#ifdef unix
	include 'pplinc/PLTCOM.DAT'
	include 'pplinc/GKSCM1.INC'
#else
	include 'pplinc:PLTCOM.DAT'
	include 'pplinc:GKSCM1.INC'
#endif
C
	CHARACTER TERM(12)*10,TYPE*10
#ifndef unix
	integer*4 lib$get_symbol,len,stat
	byte FIRSTB(15,12),SECNDB(15,12),FIRST(15),SECND(15)
#else
	INTEGER FIRSTB(15,12),SECNDB(15,12),FIRST(15),SECND(15)
#endif
	INTEGER ITERM(12),FIRSTL(12),SECNDL(12)
	LOGICAL FRST
	DATA TERM/'VT240','GVT+','ZENITH','TEK4010','TEK41XX','MAC',
     *		  'TAB','TEK4105','TEK4014','HIREZ','HP2397','GP220'/
	DATA ITERM/1,2,3,4,5,4,6,7,4,8,9,10/,FRST/.TRUE./,NTERM/12/
C
	DATA FIRSTB/27,91,63,51,56,104,9*0,
     *		    28,27,12,12*0,
     *		    27,91,50,74,11*0,
     *		    27,12,13*0,
     *		    27,37,33,48,27,76,86,48,27,83,69,49,3*0,
     *		    29,27,12,12*0,
     *		    27,37,33,48,27,76,86,48,27,12,5*0,
     *		    27,91,50,74,27,49,9*0,
     *		    27,42,100,102,27,42,100,99,27,91,51,56,104,27,12,
     *		    15*0,
     *		    15*0,
     *              15*0/
	DATA SECNDB/27,91,63,51,56,108,9*0,
     *		    24,14*0,
     *		    27,12,27,58,56,102,9*0,
     *		    27,12,13*0,
     *		    27,83,75,33,27,76,86,49,27,37,33,49,3*0,
     *		    27,12,27,34,48,103,9*0,
     *		    27,12,27,76,86,49,27,37,33,49,5*0,
     *		    27,12,27,50,11*0,
     *		    27,91,51,56,108,27,42,100,100,27,42,100,101,2*0,
     *		    15*0,
     *		    15*0,
     *              15*0/
	DATA FIRSTL/6,3,4,2,12,3,10,6,15,3*0/
	DATA SECNDL/6,1,6,2,12,6,10,4,13,3*0/
C
C
C     SET THE PLOTTING MODE.
C
      PMODE = ALPHA
C
C     IF THE LAST POINT WAS NOT PLOTTED BECAUSE THE PEN WAS UP,
C     PLOT THE POINT NOW.
C
      IF(SAVEF)CALL XYZPLT
C
C     CHECK THE PLOT TYPE.
C
      GO TO(10,50,70,10,10,20,20),PTYPE + 3
C
C     CHECK THE TEKTERMINAL TYPE.
C
   10 IF(TTYPE .EQ. -4662 .OR. TTYPE .EQ. -4663)GO TO 40
C
C     MOVE TO HOME POSITION AND ERASE THE SCREEN ON OTHER TEK-
C     TRONICS GRAPHICS TERMINALS.
C
	IF(FRST)THEN
#ifdef unix
	call getenv('GRAPHTERM',TYPE)
	LENS=lnblk(type,10)
	IF(lens.le.0)then
#else
        STAT=LIB$GET_SYMBOL('GRAPHTERM',TYPE,LEN)
        LENS=LEN
        IF(.NOT.STAT)THEN
#endif
	ITYPE=1
	ELSE
	DO 14 I=1,NTERM
	IF(TYPE(1:LENS).EQ.TERM(I))GOTO 11
14	CONTINUE
	I=1
11	ITYPE=ITERM(I)
	ENDIF
	DO 12 I=1,15
12	FIRST(I)=FIRSTB(I,ITYPE)
	DO 13 I=1,15
13	SECND(I)=SECNDB(I,ITYPE)
	FRST=.FALSE.
	ENDIF
C
	IF(HPSCLA.GT.700.)THEN
	CALL CHOUT(FIRST,FIRSTL(ITYPE))
	HPSCLA=100.
	ELSE
	CALL CHOUT(SECND,SECNDL(ITYPE))
	HPSCLA=779.
	ENDIF
C
C     TRANSMIT SYNC CHARACTERS WHILE THE PHOSPHORS IN THE SCREEN
C     COOL DOWN.
C
        NDELAY = IFIX(0.7 * FLOAT(BAUDR / 10))
        DO 30 J=1,NDELAY
   30   CALL CHOUT(SYN,1)
      CALL CHDMP
      GO TO 60
C
C     CHECK THE GRAPHICS FLAG.
C
   40 IF(.NOT. GRAPHF)THEN
C
C     TRANSMIT THE 'PLOTTER ON' COMMAND TO THE TEKTRONIX PLOTTER.
C
        CALL CHOUT(TEKON,3)
      ENDIF
C
C     MOVE TO HOME POSITION ON THE TEKTRONIX PLOTTER.
C
      CALL CHOUT(HOME,5)
C
C     SELECT ALPHANUMERIC MODE ON THE TEKTRONIX PLOTTER.
C
      CALL CHOUT(US,1)
C
C     TRANSMIT THE 'PLOTTER OFF' COMMAND TO THE TEKTRONIX PLOTTER.
C
      CALL CHOUT(TEKOFF,5)
      CALL CHDMP
C
C     RESET THE GRAPHICS FLAG.
C
60    GRAPHF = .FALSE.
C
C     CHECK FOR OTHER PLOTTING DEVICES.
C
	GOTO 25
C
C	GKS ERASE
C
20	CALL PPLGFLUSH
	IF(HPSCLA.GT.700.)THEN
	    HPSCLA=100.
#ifdef core
	    if(.not.gksopn)then
		call crinit
		gksopn=.true.
	    endif
	    call delallretainsegs()
	    call newframe()
	    call createretainseg(1)
#else
	    CALL GCLRWK(WSID,0)
#endif
	ELSE
	    HPSCLA=779.
	ENDIF
25      GOTO(50,90,90,90,70,90,70),PTYPE + 3
C
C     CHECK THE PLOTTING FLAG.
C
   70 IF(PLTFLG)THEN
C
C     MOVE TO A NEW PLOTTING AREA AND RESET THE ZETA PLOTTER.
C
        PEN = .FALSE.
        CALL ZABMV
C
C     WRITE OUT THE PLOTTING BUFFER.
C
        CALL BINFSH
        PLTFLG = .FALSE.
      ENDIF
	GOTO 90
50	IF(PLTFLG)THEN
C
C	RESET THE HP PLOTTER
C
	A=0.
	B=0.
	PEN=.FALSE.
	CALL HPABMV
	CALL ZBUFFT('SP0;',4)
	OLDHTA=0
	CALL HFLUSH
	PLTFLG=.FALSE.
	ENDIF
C
C     CHECK THE PLOTTING FLAG.
C
   90 LPEN = .FALSE.
C
C     INITIALIZE THE DATA VARIABLES FOR THE NEXT PLOT.
C
      A = 0.0
      B = 0.0
      AOLD = 0.0
      BOLD = 0.0
      COLD = 0.0
C
	IF(TTYPE.GT.0)THEN
	IF(ASIZE / TWIDTH .GT. BSIZE / THIGHT)THEN
	PLTOTA = TSCLA /ASIZE
	PLTOTB =TSCLB /ASIZE * TWIDTH /THIGHT
      ELSE
        PLTOTA = TSCLA / BSIZE * THIGHT / TWIDTH
        PLTOTB = TSCLB / BSIZE
      ENDIF
	ELSE
	PLTOTA=TSCLA/(TWIDTH*FACTOR)
	PLTOTB=TSCLB/(THIGHT*FACTOR)
	ENDIF
C
C     RECOMPUTE THE HP PLOTTER CONVERSION FACTORS.
C
C
C     RESTORE THE DEFAULT MINIMUM AND MAXIMUM ALLOWABLE VALUES OF
C     A AND B.
C
      AMIN = 0.0
      BMIN = 0.0
      AMAX = ASIZE
      BMAX = BSIZE
C
C     CALCULATE THE OFF-SCREEN FLAG.
C
      LOSFLG = 0
      IF(AOLD - 0.001 .GT. AMAX)LOSFLG = 1
      IF(AOLD + 0.001 .LT. AMIN)LOSFLG = 2
      IF(BOLD - 0.001 .GT. BMAX)LOSFLG = LOSFLG + 4
      IF(BOLD + 0.001 .LT. BMIN)LOSFLG = LOSFLG + 8
C
C     TRANSFORM THE ABSOLUTE ORIGIN TO THE USER'S COORDINATE SYS-
C     TEM.
C
      CALL TFORMI(XOLD,YOLD,ZOLD)
      IF(WINDOF)THEN
C
C     CALCULATE THE OFF-WINDOW FLAG.
C
        LOWFLG = 0
        IF(XOLD .GT. XMAX)LOWFLG = 1
        IF(XOLD .LT. XMIN)LOWFLG = 2
        IF(YOLD .GT. YMAX)LOWFLG = LOWFLG + 4
        IF(YOLD .LT. YMIN)LOWFLG = LOWFLG + 8
      ENDIF
      RETURN
C
      END
