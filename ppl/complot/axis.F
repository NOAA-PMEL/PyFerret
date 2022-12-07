      SUBROUTINE AXIS(XALOW,XAHIGH,XAORG,YALOW,YAHIGH,YAORG,XATIC,
     * YATIC,NATICX,NATICY)
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
C**    @(#)axis.f	1.1    3/10/88
*
* 11/2022 *acm* Code cleanup: Modernize DO-loop syntax. 
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
C     CALL AXIS(XLOW,XHIGH,XORG,YLOW,YHIGH,YORG,XTIC,YTIC,NSTICX,
C    * NSTICY)
C
C     THIS SUBROUTINE DRAWS A SET OF X AND Y AXES IN THE PLOTTING
C     AREA.  BOTH LARGE AND SMALL TICK MARKS MAY BE DRAWN ON THE
C     AXES AT REGULAR INTERVALS.
C
C     THE VARIOUS PARAMETERS AND THEIR DEFINITIONS ARE LISTED
C     BELOW:
C        XLOW   - MINIMUM VALUE OF THE X-AXIS IN USER'S UNITS.
C        XHIGH  - MAXIMUM VALUE OF THE X-AXIS IN USER'S UNITS.
C        XORG   - VALUE OF X AT WHICH THE Y-AXIS CROSSES THE
C                 X-AXIS IN USER'S UNITS.
C        YLOW   - MINIMUM VALUE OF THE Y-AXIS IN USER'S UNITS.
C        YHIGH  - MAXIMUM VALUE OF THE Y-AXIS IN USER'S UNITS.
C        YORG   - VALUE OF Y AT WHICH THE X-AXIS CROSSES THE
C                 Y-AXIS IN USER'S UNITS.
C        XTIC   - DISTANCE BETWEEN LARGE TICK MARKS ON THE X-AXIS
C                 IN USER'S UNITS.
C        YTIC   - DISTANCE BETWEEN LARGE TICK MARKS ON THE Y-AXIS
C                 IN USER'S UNITS.
C        NSTICX - NUMBER OF SMALL TICK MARKS BETWEEN EACH OF THE
C                 LARGE TICK MARKS ON THE X-AXIS.
C        NSTICY - NUMBER OF SMALL TICK MARKS BETWEEN EACH OF THE
C                 LARGE TICK MARKS ON THE Y-AXIS.
C
C     AXIS MAKES SEVERAL CHECKS ON THE PARAMETERS TO MAKE SURE
C     THAT THEY ARE VALID.  THEY ARE ALTERED INTERNALLY IF NECES-
C     SARY.
C
C
C     TRANSFER THE PARAMETERS TO NEW VARIABLES.
C
      XLOW = XALOW
      XHIGH = XAHIGH
      XORG = XAORG
      YLOW = YALOW
      YHIGH = YAHIGH
      YORG = YAORG
      XTIC = ABS(XATIC)
      YTIC = ABS(YATIC)
      NSTICX = NATICX
      NSTICY = NATICY
C
C     CHECK THE PARAMETERS TO MAKE SURE THAT THEY ARE VALID.
C
      IF(XLOW .GT. XHIGH)THEN
        TEMP = XLOW
        XLOW = XHIGH
        XHIGH = TEMP
      ENDIF
      IF(YLOW .GT. YHIGH)THEN
        TEMP = YLOW
        YLOW = YHIGH
        YHIGH = YLOW
      ENDIF
      IF(XORG .LT. XLOW)XORG = XLOW
      IF(YORG .LT. YLOW)YORG = YLOW
      IF(XORG .GT. XHIGH)XORG = XHIGH
      IF(YORG .GT. YHIGH)YORG = YHIGH
      IF(NSTICX .LT. 0)NSTICX = 0
      IF(NSTICY .LT. 0)NSTICY = 0
C
C     SET VECTRS MODE.
C
      CALL VECTRS
C
C     START AT THE ORIGIN.
C
      X = XORG
      Y = YORG
      CALL PLOT(X,Y,0,0)
C
C     IF THERE ARE NO TICK MARKS ON THE Y-AXIS, DRAW THE ENTIRE
C     AXIS.
C
      IF(YTIC .EQ. 0.0)THEN
        CALL PLOT(X,YLOW,0,0)
        CALL PLOT(X,YHIGH,1,0)
        CALL PLOT(X,Y,0,0)
      ELSE
C
        SMY = YTIC / FLOAT(NSTICY + 1)
C
        IF(YLOW .LT. YORG)THEN
C
C     DRAW THE NEGATIVE Y-AXIS.
C
   10     IF(NSTICY .EQ. 0)GO TO 30
          DO 20 J=1,NSTICY
            Y = Y - SMY
            IF(Y .LT. YLOW)GO TO 40
            CALL PLOT(X,Y,1,5)
   20     CONTINUE
   30     Y = Y - SMY
          IF(Y .LT. YLOW)GO TO 40
          CALL PLOT(X,Y,1,6)
          GO TO 10
   40     CALL PLOT(X,YLOW,1,0)
          Y = YORG
          CALL PLOT(X,Y,0,0)
        ENDIF
C
        IF(YHIGH .GT. YORG)THEN
C
C     DRAW THE POSITIVE Y-AXIS.
C
   50     IF(NSTICY .EQ. 0)GO TO 70
          DO 60 J=1,NSTICY
            Y = Y + SMY
            IF(Y .GT. YHIGH)GO TO 80
            CALL PLOT(X,Y,1,5)
   60	  CONTINUE     
   70     Y = Y + SMY
          IF(Y .GT. YHIGH)GO TO 80
          CALL PLOT(X,Y,1,6)
          GO TO 50
   80     CALL PLOT(X,YHIGH,1,0)
          Y = YORG
          CALL PLOT(X,Y,0,0)
        ENDIF
      ENDIF
C
C     IF THERE ARE NO TICK MARKS ON THE X-AXIS, DRAW THE ENTIRE
C     AXIS.
C
      IF(XTIC .EQ. 0.0)THEN
        CALL PLOT(XLOW,Y,0,0)
        CALL PLOT(XHIGH,Y,1,0)
        CALL PLOT(X,Y,0,0)
      ELSE
C
        SMX = XTIC / FLOAT(NSTICX + 1)
C
        IF(XLOW .LT. XORG)THEN
C
C     DRAW THE NEGATIVE X-AXIS.
C
   90     IF(NSTICX .EQ. 0)GO TO 110
          DO 100 J=1,NSTICX
            X = X - SMX
            IF(X .LT. XLOW)GO TO 120
            CALL PLOT(X,Y,1,7)
  100     CONTINUE
  110     X = X - SMX
          IF(X .LT. XLOW)GO TO 120
          CALL PLOT(X,Y,1,8)
          GO TO 90
  120     CALL PLOT(XLOW,Y,1,0)
          X = XORG
          CALL PLOT(X,Y,0,0)
        ENDIF
C
        IF(XHIGH .GT. XORG)THEN
C
C     DRAW THE POSITIVE X-AXIS.
C
  130     IF(NSTICX .EQ. 0)GO TO 150
          DO 140 J=1,NSTICX
            X = X + SMX
            IF(X .GT. XHIGH)GO TO 160
            CALL PLOT(X,Y,1,7)
  140     CONTINUE
  150     X = X + SMX
          IF(X .GT. XHIGH)GO TO 160
          CALL PLOT(X,Y,1,8)
          GO TO 130
  160     CALL PLOT(XHIGH,Y,1,0)
          X = XORG
          CALL PLOT(X,Y,0,0)
        ENDIF
      ENDIF
C
C     THE AXES ARE COMPLETE.
C
      RETURN
C
      END
