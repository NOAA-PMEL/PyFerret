      SUBROUTINE RANGE_DM(uamin,uamax,udel,uminr,umaxr,dint)
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
C**    @(#)range.f	1.1    3/10/88
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
C     CALL RANGE_DM(umin,umax,udel,uminr,umaxr,dint)
C
C     This subroutine finds two values for a variable such that
C     the range between them is evenly divisible into approxi-
C     mately a specified number of intervals and contains the
C     minimum and maximum values of the variable.
C     Based on RANGE, but computed for output as degrees and minutes.
C
C     The various parameters and their definitions are listed
C     below:
C        UMIN   - previously computed minimum value of the range.
C        UMAX   - previously computed minimum value of the range.
C        UDEL  -  previous delta value.
C        UMINR  - new minimum value of the range.
C        UMAXR  - new maximum value of the range.
C        DINT   - new interval size.
C
C  Calling argument declarations
      REAL uamin, uamax, udel, uminr, umaxr, dint

C Local variable declarations

      REAL umin, umax, temp, fm
      INTEGER m
C
C     Transfer the parameters to new variables.
C
      umin = uamin
      umax = uamax
C
C     Check the parameters to make sure that they are valid.
C
      IF (umin .GT. umax) THEN
        temp = umin
        umin = umax
        umax = temp
      ENDIF
C
C
C     Change to intervals appropriate for deg/minute formatting
C     If the previously computed interval is less than 1 minute,
C     make it 1 minute; change other intervals to correspond to
C     2, 5, 10, 30 minutes, or whole degrees.
C
      IF (udel .LE. 0.01) THEN
         dint = 1./120.   ! half minute 0.0083 deg
      ELSE IF (udel .LE. 0.02) THEN
         dint = 1./60.   ! one minute 0.01667 deg
      ELSE IF (udel .LE. 0.05) THEN
         dint = 2./60.   ! two minutes 0.0333 deg
      ELSE IF (udel .LE. 0.1) THEN 
         dint = 5./60.   ! five minutes 0.0833 deg
      ELSE IF (udel .LE. 0.2) THEN
         dint = 10./60.   ! ten minutes 0.1667 deg
      ELSE IF (udel .LE. 0.5) THEN
         dint = 15./60.   ! 15 minutes 0.25 deg
      ELSE
         dint = udel
      ENDIF
C
C     recalculate the minimum value of the range.
C
      fm = umin / dint
      m = IFIX(fm)
      IF (fm .LT. 0.0) m = m - 1
      uminr = dint * FLOAT(m)
C
C     recalculate the maximum value of the range.
C
      fm = umax / dint
      m = IFIX (fm)
      IF (fm .GT. 0.0) m = m + 1
      umaxr = dint * FLOAT(m)
C
      RETURN
      END
