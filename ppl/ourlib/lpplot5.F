      SUBROUTINE LPPLOT(XARRAY,YARRAY,NPTS,XFACT,YFACT,XLOW,YLOW,
     1    XORG,YORG,XTIC,YTIC,CHAR,LUN,NLINES)
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
C**    @(#)lpplot5.f	1.1    3/10/88
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
C         LPPLOT GENERATES A LINE PRINTER PLOT 101 COLUMNS WIDE BY
C         NLINES LINES HIGH. IT DOES NOT DO A PAGE EJECT BEFORE HAND NOR
C         DOES IT DO AXES LABELLING.  IT WILL LABEL TICK MARKS HOWEVER.
C
C         THE CALLING PARAMETERS ARE SIMILAR TO COMPLOT:
C                 XARRAY:HOLDS THE X VALUE OF THE POINTS.
C                 YARRAY:HOLDS THE Y VALUE OF THE POINTS.
C                 NPTS:  THE NUMBER OF POINTS TO BE PLOTTED.
C                 XFACT: NUMBER OF COLUMNS PER USER UNIT.
C                 YFACT: NUMBER OF LINES PER USER UNIT.
C                 XLOW:  LOWEST VALUE OF THE X AXIS.
C                 YLOW:  LOWEST VALUE OF THE Y AXIS.
C                 XORG:  THE X-COORDINATE WHERE THE AXES CROSS.
C                 YORG:  THE Y-COORDINATE WHERE THE AXES CROSS.
C                 XTIC:  THE DISTANCE BETWEEN X TICK MARKS IN
C                        USER UNITS.
C                 YTIC:  THE DISTANCE BETWEEN Y TICK MARKS IN
C                        USER UNITS.
C                 CHAR:  THE CHARACTER TO BE USED TO MARK A POINT.
C                 LUN:   THE LOGICAL UNIT NUMBER ON WHICH TO PUT THE PLOT.
C                 NLINES:HEIGHT OF THE PLOT IN LINES.
C
      DIMENSION XARRAY(1),YARRAY(1)

#ifdef STAR_1_SUPPORTED
      integer*1 DATBUF(101),PRBUFF(121),BLANK,STAR,EYE,CHAR
#else
      BYTE DATBUF(101),PRBUFF(121),BLANK,STAR,EYE,CHAR
#endif

      DIMENSION XLAB(20),LABLOC(20)
      EQUIVALENCE(PRBUFF(11),DATBUF(1))
      DATA BLANK/1H /,STAR/1H*/,EYE/1HI/

      YINCD2=0.5/YFACT
      LINORG=FLOAT(NLINES)+0.5-(YORG-YLOW)*YFACT
      NXTICS=100.0/(XFACT*XTIC)+1

      DO 700 LINE=1,NLINES

      DO 50 JCOL=1,121
   50 PRBUFF(JCOL)=BLANK
      CALL XFILL(XORG,DATBUF,XFACT,XLOW,STAR)

      IF(LINE.NE.LINORG)GO TO 200
      DO 80 JCOL=1,101
   80 DATBUF(JCOL)=STAR
      DO 100 JTIC=1,NXTICS
      XT=(JTIC-1)*XTIC+XLOW
      JXT=(XT-XLOW)*XFACT+1.5
      IF(JXT.GT.101)GO TO 200
      DATBUF(JXT)=EYE
      XLAB(JTIC)=XT
      LABLOC(JTIC)=JXT+5
  100 CONTINUE

  200 IF(LINE.NE.LINORG+1)GO TO 250
      DO 230 JTIC=1,NXTICS
      L=LABLOC(JTIC)
      IF(L.LT.1 .OR. L.GT.111)GO TO 230
      CALL CODEA1(XLAB(JTIC),PRBUFF(L))
  230 CONTINUE

  250 YCENT=(NLINES-LINE)/YFACT+YLOW
      DO 400 JVAL=1,NPTS
      IF(ABS(YCENT-YARRAY(JVAL)).GT.YINCD2)GO TO 400
      CALL XFILL(XARRAY(JVAL),DATBUF,XFACT,XLOW,CHAR)
  400 CONTINUE

      IF(AMOD(YCENT-YLOW,YTIC).LE.YINCD2)GO TO 500
      WRITE(LUN,2000)PRBUFF
 2000 FORMAT(11X,121A1)
      GO TO 700

  500 WRITE(LUN,3000)YCENT,PRBUFF
 3000 FORMAT(1X,F10.2,121A1)

  700 CONTINUE

      RETURN
C**  THIS PROGRAM VALID ON FTN4 AND FTN5 **
      END
C*F45V1P0*

      SUBROUTINE XFILL(X,DATBUF,XFACT,XLOW,CHAR)

#ifdef STAR_1_SUPPORTED
      integer*1 DATBUF(101),CHAR
#else
      BYTE DATBUF(101),CHAR
#endif

      JCOL=(X-XLOW)*XFACT+1.5
      IF(JCOL.LT.1 .OR. JCOL.GT.101)RETURN
      DATBUF(JCOL)=CHAR
      RETURN
C**  THIS PROGRAM VALID ON FTN4 AND FTN5 **
      END
C*F45V1P0*

      SUBROUTINE CODEA1(X,COBUFF)

#ifdef STAR_1_SUPPORTED
      integer*1 COBUFF(1)
#else
      BYTE COBUFF(1)
#endif

      CHARACTER STR*8
      WRITE(STR,'(F8.1)')X
      READ(STR,999)(COBUFF(I),I=1,8)
999	FORMAT(8A1)
      RETURN
C**  THIS PROGRAM VALID ON FTN4 AND FTN5 **
      END
