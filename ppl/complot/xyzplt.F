      SUBROUTINE XYZPLT

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
****************************************************************
C revised for FERRET
C from PPLUS :
C	Directory PMEL:[PLOT5.LIBRARY.COMPLOT]
C	ABPLT.FOR;2               9   5-FEB-1985 11:33 
C search for "FERRET" to find all revisions
C *sh* 9/19/88 - rev 1.0 for FERRET 1.10+
C	- if GKS (ptype>2) then bypass clipping at screen limits
C !!! ALSO :  Routine overflows subscripts at label 70
C	      70 XINTER = XMXMN(IAND(LOWFLG,3))
C	      logic to deal with a pair of points whose connecting segment
C		crosses the corner of the window is wrong
C	      ...bounds checking is OFF to match Denbo's compile stmt
****************************************************************
C     CALL XYZPLT

C     THIS SUBROUTINE PERFORMS WINDOW CLIPPING IAND THEN MOVES THE
C     PEN FROM THE CURRENT POSITION TO A NEW POINT.



******************************************************************
C FERRET mod - for explicit declarations - 9-19-88
C      DIMENSION XMXMN(2),YMXMN(2)
      REAL XMXMN(2),YMXMN(2)
      EQUIVALENCE (XMXMN(1),XMAX),(YMXMN(1),YMAX)
      REAL xinter, yinter, zinter, dx, dy, dz, cosx, cosy, cosz, dist

#ifdef unix
      include 'pltcom_dat.decl'
      include 'PLTCOM.DAT'
#else
      INCLUDE 'tmap_pplv11inc:tmap_PLTCOM.DAT'
#endif
C end of FERRET mod
******************************************************************
C     CHECK THE PLOTTING MODE.

      IF(PMODE .EQ. ALPHA)GO TO 30

C     CHECK THE PEN POSITION.

      IF(PENF)GO TO 10

C     TEMPORARILY STORE THE NEW POSITION.

      XNEW = XTEMP
      YNEW = YTEMP
      ZNEW = ZTEMP
      SAVEF = .TRUE.
      RETURN

C     CHECK THE PLOTTING MODE.

  10  IF(PMODE .NE. POINT)GO TO 20
      XNEW = XTEMP
      YNEW = YTEMP
      ZNEW = ZTEMP
      GO TO 30

C     IF THE LAST POINT WAS NOT PLOTTED BECAUSE THE PEN WAS UP,
C     PLOT THE POINT NOW.

  20  IF(.NOT. SAVEF)GO TO 60
      PENF = .FALSE.
  30  IF(.NOT. WINDOF)GO TO 40

C     CALCULATE THE NEW OFF-WINDOW FLAG.

      OWFLG = 0
      IF(XNEW .GT. XMAX)OWFLG = 1
      IF(XNEW .LT. XMIN)OWFLG = 2
      IF(YNEW .GT. YMAX)OWFLG = OWFLG + 4
      IF(YNEW .LT. YMIN)OWFLG = OWFLG + 8

C     CHECK THE NEW POINT TO SEE IF IT IS WITHIN THE BOUNDARY OF
C     THE WINDOW.  IF NOT, DO NOT PLOT IT.

      IF(OWFLG .NE. 0)GO TO 50

C     TRANSFORM THE POINT TO THE SCREEN COORDINATE SYSTEM.

  40  CALL TFORM(XNEW,YNEW,ZNEW)

C     PLOT TO THE POINT.

      CALL ABPLT

C     MOVE THE COORDINATES OF THE NEW POINT TO THE OLD POINT.

  50  XOLD = XNEW
      YOLD = YNEW
      ZOLD = ZNEW
      LOWFLG = OWFLG
      SAVEF = .FALSE.

C     CHECK THE PLOTTING MODE.

      IF(PMODE .EQ. POINT .OR. PMODE .EQ. ALPHA)RETURN
      PENF = .TRUE.
      IF(PMODE .NE. DASH)GO TO 60

C     REINITIALIZE THE SEGMENT COUNTER.

      IDSEG = 1
      SEG = DSEG(1)
  60  XNEW = XTEMP
      YNEW = YTEMP
      ZNEW = ZTEMP
      IF(.NOT. WINDOF)GO TO 120

C     CALCULATE THE NEW OFF-WINDOW FLAG.

      OWFLG = 0
      IF(XNEW .GT. XMAX)OWFLG = 1
      IF(XNEW .LT. XMIN)OWFLG = 2
      IF(YNEW .GT. YMAX)OWFLG = OWFLG + 4
      IF(YNEW .LT. YMIN)OWFLG = OWFLG + 8

C     CHECK THE OLD AND NEW POINTS TO SEE IF THEY ARE BOTH OUT-
C     SIDE THE BOUNDARY OF THE WINDOW IN THE SAME DIRECTION.  IF
C     SO, NO INTERCEPT EXISTS.

C	RSX-11M

C      IF(IIAND(LOWFLG,OWFLG) .NE. 0)GO TO 190

C	END RSX-11M
C	VAX/VMS

#ifdef unix
# ifdef NEED_IAND
       IF(IAND(LOWFLG,OWFLG) .NE. 0)GOTO 190
# else
       IF(AND(LOWFLG,OWFLG) .NE. 0)GOTO 190
# endif
#else
       IF(IAND(LOWFLG,OWFLG) .NE. 0)GOTO 190
#endif

C	END VAX/VMS


C     CHECK THE OLD POINT TO SEE IF IT IS WITHIN THE BOUNDARY OF
C     THE WINDOW.  IF NOT, COMPUTE THE INTERCEPT.

      IF(LOWFLG .EQ. 0)GO TO 90

C     CHECK THE HORIZONTAL LIMITS OF THE WINDOW.

#ifdef unix
# ifdef NEED_IAND
      IF(ISHFT(LOWFLG,-2) .EQ. 0)GO TO 70
# else
      IF(rSHIFT(LOWFLG,2) .EQ. 0)GO TO 70
# endif
#else
      IF(ISHFT(LOWFLG,-2) .EQ. 0)GO TO 70
#endif

C     THE POINT IS OUTSIDE THE HORIZONTAL LIMITS OF THE WINDOW.
C     CALCULATE THE INTERCEPT.

#ifdef unix
# ifdef NEED_IAND
      YINTER = YMXMN(ISHFT(LOWFLG,-2))
# else
      YINTER = YMXMN(rSHIFT(LOWFLG,2))
# endif
#else
      YINTER = YMXMN(ISHFT(LOWFLG,-2))
#endif

      ZINTER = (ZNEW - ZOLD) * (YINTER - YOLD) / (YNEW - YOLD) +
     * ZOLD
      XINTER = (XNEW - XOLD) * (YINTER - YOLD) / (YNEW - YOLD) +
     * XOLD

C     CHECK THE INTERCEPT TO INSURE THAT IT IS ON THE BOUNDARY
C     OF THE WINDOW

      IF(XINTER + 0.001 .GE. XMIN .AND. XINTER - 0.001 .LE. XMAX)
     * GOTO 80

C     THE POINT IS OUTSIDE THE VERTICAL LIMITS OF THE WINDOW.
C     CALCULATE THE INTERCEPT.

#ifdef unix
# ifdef NEED_IAND
 70   XINTER = XMXMN(IAND(LOWFLG,3))
# else
 70   XINTER = XMXMN(AND(LOWFLG,3))
# endif
#else
 70   XINTER = XMXMN(IAND(LOWFLG,3))
#endif 

      YINTER = (YNEW - YOLD) * (XINTER - XOLD) / (XNEW - XOLD) +
     * YOLD
      ZINTER = (ZNEW - ZOLD) * (XINTER - XOLD) / (XNEW - XOLD) +
     * ZOLD

C     CHECK THE INTERCEPT TO INSURE THAT IT IS ON THE BOUNDARY
C     OF THE WINDOW.

      IF(YINTER + 0.001 .LT. YMIN .OR. YINTER - 0.001 .GT. YMAX)
     * GO TO 190

C     TRANSFORM THE INTERCEPT TO THE SCREEN COORDINATE SYSTEM.

  80  CALL TFORM(XINTER,YINTER,ZINTER)
      XOLD = XINTER
      YOLD = YINTER
      ZOLD = ZINTER
      AOLD = ANEW
      BOLD = BNEW 

C     CALCULATE THE OLD OFF-SCREEN FLAG.

C     LOSFLG = 0
******************************************************************
C FERRET mod - bypass GKS clipping - 9-19-88
      IF ( ptype .LT. 3 ) THEN
          IF(AOLD - 0.001 .GT. AMAX)LOSFLG = 1
          IF(AOLD + 0.001 .LT. AMIN)LOSFLG = 2
          IF(BOLD - 0.001 .GT. BMAX)LOSFLG = LOSFLG + 4
          IF(BOLD + 0.001 .LT. BMIN)LOSFLG = LOSFLG + 8
       ENDIF
C end of FERRET mod
******************************************************************

C     CHECK THE NEW POINT TO SEE IF IT IS WITHIN THE BOUNDARY OF
C     THE WINDOW.  IF NOT, COMPUTE THE INTERCEPT.

 90   IF(OWFLG .EQ. 0)GO TO 120

C     CHECK THE HORIZONTAL LIMITS OF THE WINDOW.

#ifdef unix
# ifdef NEED_IAND
      IF(ISHFT(OWFLG,-2) .EQ. 0)GO TO 100
# else
      IF(rSHIFT(OWFLG,2) .EQ. 0)GO TO 100
# endif
#else
      IF(ISHFT(OWFLG,-2) .EQ. 0)GO TO 100
#endif

C     THE POINT IS OUTSIDE THE HORIZONTAL LIMITS OF THE WINDOW.
C     CALCULATE THE INTERCEPT.

#ifdef unix
# ifdef NEED_IAND
      YINTER = YMXMN(ISHFT(OWFLG,-2))
# else
      YINTER = YMXMN(rSHIFT(OWFLG,2))
# endif
#else
      YINTER = YMXMN(ISHFT(OWFLG,-2))
#endif
      ZINTER = (ZNEW - ZOLD) * (YINTER - YOLD) / (YNEW - YOLD) +
     * ZOLD
      XINTER = (XNEW - XOLD) * (YINTER - YOLD) / (YNEW - YOLD) +
     * XOLD

C     CHECK THE INTERCEPT TO INSURE THAT IT IS ON THE BOUNDARY
C     OF THE WINDOW.

      IF(XINTER + 0.001 .GE. XMIN .AND. XINTER - 0.001 .LE. XMAX)
     * GO TO 110

C     THE POINT IS OUTSIDE THE VERTICAL LIMITS OF THE WINDOW.
C     CALCULATE THE INTERCEPT.
#ifdef unix
# ifdef NEED_IAND
 100  XINTER = XMXMN(IAND(OWFLG,3))
# else
 100  XINTER = XMXMN(AND(OWFLG,3))
# endif
#else
 100  XINTER = XMXMN(IAND(OWFLG,3))
#endif
      YINTER = (YNEW - YOLD) * (XINTER - XOLD) / (XNEW - XOLD) +
     * YOLD
      ZINTER = (ZNEW - ZOLD) * (XINTER - XOLD) / (XNEW - XOLD) +
     * ZOLD

C     CHECK THE PLOTTING MODE.

 110  IF(PMODE .EQ. VECTOR)GO TO 160
      DX = XINTER - XOLD
      DY = YINTER - YOLD
      DZ = ZINTER - ZOLD
      GO TO 130

C     CHECK THE PLOTTING MODE.

 120  IF(PMODE .EQ. VECTOR)GO TO 170
      DX = XNEW - XOLD
      DY = YNEW - YOLD
      DZ = ZNEW - ZOLD

C     INITIALIZE VARIABLES IN PREPARATION FOR DRAWING A DASHED
C     LINE.

 130  DIST = SQRT(DX * DX + DY * DY + DZ * DZ)
      IF(DIST .EQ. 0.0)GO TO 150
      COSX = DX / DIST
      COSY = DY / DIST
      COSZ = DZ / DIST

C     DRAW A DASHED LINE ONE SEGMENT AT A TIME.

#ifdef unix
# ifdef NEED_IAND
 140  PENF = IAND(IDSEG,1) .NE. 0
# else 
 140  PENF = AND(IDSEG,1) .NE. 0
# endif
#else
 140  PENF = IAND(IDSEG,1) .NE. 0
#endif
      IF(SEG .GE. DIST)GO TO 150
      XOLD = XOLD + SEG * COSX
      YOLD = YOLD + SEG * COSY
      ZOLD = ZOLD + SEG * COSZ

C     TRANSFORM THE END OF THE SEGMENT TO THE SCREEN COORDINATE
C     SYSTEM.

      CALL TFORM(XOLD,YOLD,ZOLD)

C     PLOT TO THE END OF THE SEGMENT.

      CALL ABPLT
      DIST = DIST - SEG
#ifdef unix
# ifdef NEED_IAND
      IDSEG = IAND(IDSEG,3) + 1
# else
      IDSEG = AND(IDSEG,3) + 1
# endif
#else
      IDSEG = IAND(IDSEG,3) + 1
#endif
      SEG = DSEG(IDSEG)
      GO TO 140
 150  SEG = SEG - DIST

C     CHECK THE NEW POINT TO SEE IF IT IS WITHIN THE BOUNDARY OF
C     THE WINDOW.  IF NOT, PLOT TO THE INTERCEPT.

      IF(OWFLG .EQ. 0)GO TO 170

C     TRANSFORM THE INTERCEPT TO THE SCREEN COORDINATE SYSTEM.

 160  CALL TFORM(XINTER,YINTER,ZINTER)
      GO TO 180

C     TRANSFORM THE POINT TO THE SCREEN COORDINATE SYSTEM.

 170  CALL TFORM(XNEW,YNEW,ZNEW)

C     PLOT TO THE POINT.

 180  CALL ABPLT

C     MOVE THE COORDINATES OF THE NEW POINT TO THE OLD POINT.

 190  XOLD = XNEW
      YOLD = YNEW
      ZOLD = ZNEW
      LOWFLG = OWFLG
      RETURN 

      END


