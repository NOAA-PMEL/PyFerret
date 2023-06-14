      SUBROUTINE ABPLT
                
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
* revised for FERRET
* from PPLUS :
*	Directory PMEL:[PLOT5.LIBRARY.COMPLOT]
*	ABPLT.FOR;2               9   5-FEB-1985 11:33 
* search for "FERRET" to find all revisions
* *sh* 9/13/88 - rev 1.0 for FERRET 1.10+
*	- if GKS (ptype>2) then bypass clipping at screen limits
* 
* *kob* 3/02 - need to add sgi to ifdef statement 
****************************************************************
*     CALL ABPLT

*     THIS SUBROUTINE PERFORMS SCREEN CLIPPING AND THEN MOVES THE
*     PEN FROM THE CURRENT POSITION TO A NEW POINT.


******************************************************************
* FERRET mod - for explicit declarations - 9-13-88
C     DIMENSION AMXMN(2),BMXMN(2)
      REAL AMXMN(2),BMXMN(2), binter, ainter
      EQUIVALENCE (AMXMN(1),AMAX),(BMXMN(1),BMAX)

      include 'pltcom_dat.decl'
      include 'PLTCOM.DAT'

C end of FERRET mod
******************************************************************

C     CALCULATE THE NEW OFF-SCREEN FLAG.

******************************************************************
C FERRET mod - bypass GKS clipping - 9-13-88
      IF ( ptype .LT. 3 ) THEN
          OSFLG = 0
          IF(ANEW - 0.001 .GT. AMAX)OSFLG = 1
          IF(ANEW + 0.001 .LT. AMIN)OSFLG = 2
          IF(BNEW - 0.001 .GT. BMAX)OSFLG = OSFLG + 4
          IF(BNEW + 0.001 .LT. BMIN)OSFLG = OSFLG + 8
       ELSE
          OSFLG = 0
          losflg = 0            ! reset by many PPLUS routines
       ENDIF
* end of FERRET mod
******************************************************************

C     CHECK THE PEN POSITION.

       IF(.NOT. PENF)GO TO 90

C     CHECK THE PLOTTING MODE.

      IF(PMODE .EQ. POINT)GO TO 80

C     CHECK THE OLD AND NEW POINTS TO SEE IF THEY ARE BOTH OUT-
C     SIDE THE BOUNDARY OF THE SCREEN IN THE SAME DIRECTION.  IF
C     SO, NO INTERCEPT EXISTS.

      IF(IAND(LOSFLG,OSFLG) .NE. 0)GO TO 100

C     CHECK THE OLD POINT TO SEE IF IT IS WITHIN THE BOUNDARY OF
C     THE SCREEN.  IF NOT, COMPUTE THE INTERCEPT.

      IF(LOSFLG .EQ. 0)GO TO 30

C     CHECK THE HORIZONTAL LIMITS OF THE SCREEN.

      if(ishft(losflg,-2) .eq. 0)goto 10

C    THE POINT IS OUTSIDE THE HORIZONTAL LIMITS OF THE SCREEN.
C    CALCULATE THE INTERCEPT.
       
      binter = bmxmn(ishft(losflg,-2))

      AINTER = (ANEW - AOLD) * (BINTER - BOLD) / (BNEW - BOLD) +
     * AOLD
    
C     CHECK THE INTERCEPT TO INSURE THAT IT IS ON THE BOUNDARY
C     OF THE SCREEN.

      IF(AINTER + 0.001 .GE. AMIN .AND. AINTER - 0.001 .LE. AMAX)
     * GO TO 20

C     THE POINT IS OUTSIDE THE VERTICAL LIMITS OF THE SCREEN.
C     CALCULATE THE INTERCEPT.

 10   ainter = amxmn(iand(losflg,3))

      BINTER = (BNEW - BOLD) * (AINTER - AOLD) / (ANEW - AOLD) +
     * BOLD

C     CHECK THE INTERCEPT TO INSURE THAT IT IS ON THE BOUNDARY
C     OF THE SCREEN.

      IF(BINTER + 0.001 .LT. BMIN .OR. BINTER - 0.001 .GT. BMAX)
     * GO TO 100

C     PLOT TO THE INTERCEPT WITH THE PEN UP.

  20  A = AINTER - AMIN
      B = BINTER - BMIN
      PEN = .FALSE.
      CALL ABMV
      GO TO 40

C     CHECK THE OLD POINT TO SEE IF IT IS WITHIN THE BOUNDARY OF
C     THE WINDOW.  IF NOT, PLOT TO THE INTERCEPT WITH THE PEN UP.

  30  IF(LOWFLG .EQ. 0)GO TO 40
      A = AOLD - AMIN
      B = BOLD - BMIN
      PEN = .FALSE.
      CALL ABMV

C      CHECK THE NEW POINT TO SEE IF IT IS WITHIN THE BOUNDARY OF
C      THE SCREEN.  IF NOT, COMPUTE THE INTERCEPT.

  40  IF(OSFLG .EQ. 0)GO TO 70

C     CHECK THE HORIZONTAL LIMITS OF THE SCREEN.

      IF(ISHFT(OSFLG,-2) .EQ. 0)GO TO 50

C     THE POINT IS OUTSIDE THE HORIZONTAL LIMITS OF THE SCREEN.
C     CALCULATE THE INTERCEPT.

      BINTER = BMXMN(ISHFT(OSFLG,-2))

      AINTER = (ANEW - AOLD) * (BINTER - BOLD) / (BNEW - BOLD) +
     * AOLD

C     CHECK THE INTERCEPT TO INSURE THAT IT IS ON THE BOUNDARY
C     OF THE SCREEN.

      IF(AINTER + 0.001 .GE. AMIN .AND. AINTER - 0.001 .LE. AMAX)
     * GO TO 60

C     THE POINT IS OUTSIDE THE VERTICAL LIMITS OF THE SCREEN.
C     CALCULATE THE INTERCEPT.

 50   AINTER = AMXMN(IAND(OSFLG,3))

      BINTER = (BNEW - BOLD) * (AINTER - AOLD) / (ANEW - AOLD) +
     * BOLD

C     PLOT TO THE INTERCEPT WITH THE PEN DOWN.

  60  A = AINTER - AMIN
      B = BINTER - BMIN
      PEN = .TRUE.
      CALL ABMV
      GO TO 100

C     PLOT TO THE POINT WITH THE PEN DOWN.

  70  A = ANEW - AMIN
      B = BNEW - BMIN
      PEN = .TRUE.
      CALL ABMV
      GO TO 100

C     CHECK THE NEW POINT TO SEE IF IT IS WITHIN THE BOUNDARY OF
C     THE SCREEN.  IF NOT, DO NOT PLOT IT.

  80  IF(OSFLG .NE. 0)GO TO 100

C     PLOT TO THE POINT WITH THE PEN UP AND THEN DRAW A DOT.

      A = ANEW - AMIN
      B = BNEW - BMIN
      PEN = .FALSE.
      CALL ABMV
      PEN = .TRUE.
      CALL ABMV
      GO TO 100

C     CHECK THE NEW POINT TO SEE IF IT IS WITHIN THE BOUNDARY OF
C     THE SCREEN.  IF NOT, DO NOT PLOT IT.

  90  IF(OSFLG .NE. 0)GO TO 100

C     PLOT TO THE POINT WITH THE PEN UP.

      A = ANEW - AMIN
      B = BNEW - BMIN
      PEN = .FALSE.
      CALL ABMV

C     MOVE THE COORDINATES OF THE NEW POINT TO THE OLD POINT.

 100  AOLD = ANEW
      BOLD = BNEW
      AOSOLD = A
      BOSOLD = B
      LOSFLG = OSFLG
      RETURN

      END
