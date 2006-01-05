      SUBROUTINE RANGE(UAMIN,UAMAX,NAINTS,UMINR,UMAXR,DINT)
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
C** V590:  1/06 *acm*  prevent values of LOG10(0.0)
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
C     CALL RANGE(UMIN,UMAX,NINTS,UMINR,UMAXR,DINT)
C
C     THIS SUBROUTINE FINDS TWO VALUES FOR A VARIABLE SUCH THAT
C     THE RANGE BETWEEN THEM IS EVENLY DIVISIBLE INTO APPROXI-
C     MATELY A SPECIFIED NUMBER OF INTERVALS AND CONTAINS THE
C     MINIMUM AND MAXIMUM VALUES OF THE VARIABLE.
C
C     THE VARIOUS PARAMETERS AND THEIR DEFINITIONS ARE LISTED
C     BELOW:
C        UMIN   - MINIMUM VALUE OF THE VARIABLE.
C        UMAX   - MAXIMUM VALUE OF THE VARIABLE.
C        NINTS  - APPROXIMATE NUMBER OF INTERVALS INTO WHICH THE
C                 RANGE IS TO BE DIVIDED.
C        UMINR  - MINIMUM VALUE OF THE RANGE.
C        UMAXR  - MAXIMUM VALUE OF THE RANGE.
C        DINT   - INTERVAL SIZE.
C
C
C     TRANSFER THE PARAMETERS TO NEW VARIABLES.
C
      UMIN = UAMIN
      UMAX = UAMAX
      NINTS = IABS(NAINTS)
C
C     CHECK THE PARAMETERS TO MAKE SURE THAT THEY ARE VALID.
C
      IF(UMIN .GT. UMAX)THEN
        TEMP = UMIN
        UMIN = UMAX
        UMAX = TEMP
      ENDIF
      IF(NINTS .EQ. 0)NINTS = 1
C
C     FIND THE APPROXIMATE SIZE OF THE INTERVAL.
C
      TEMP = (UMAX - UMIN) / FLOAT(NINTS)
      IF(TEMP .EQ. 0.0)TEMP = UMAX
C
C     SCALE THE INTERVAL SIZE BY POWERS OF TEN TO A VALUE BETWEEN
C     ONE AND TEN.
C
      IF (TEMP .EQ. 0.0) THEN
         NT = 1
      ELSE
         NT = IFIX(ALOG10(TEMP))
      ENDIF
      IF(TEMP .LT. 1.0)NT = NT - 1
      POW = 10.0 ** NT
      TEMP = TEMP / POW
C
C     FIND THE CLOSEST PERMISSIBLE VALUE FOR THE INTERVAL SIZE.
C
      IF(TEMP .LT. SQRT(2.0))THEN
        DINT = POW
      ELSE IF(TEMP .LT. SQRT(10.0))THEN
        DINT = 2.0 * POW
      ELSE IF(TEMP .LT. SQRT(50.0))THEN
        DINT = 5.0 * POW
      ELSE
        DINT = 10.0 * POW
      ENDIF
C
C     CALCULATE THE MINIMUM VALUE OF THE RANGE.
C
      FM = UMIN / DINT
      M = IFIX(FM)
      IF(FM .LT. 0.0)M = M - 1
      UMINR = DINT * FLOAT(M)
C
C     CALCULATE THE MAXIMUM VALUE OF THE RANGE.
C
      FM = UMAX / DINT
      M = IFIX(FM)
      IF(FM .GT. 0.0)M = M + 1
      UMAXR = DINT * FLOAT(M)
      RETURN
C
      END
