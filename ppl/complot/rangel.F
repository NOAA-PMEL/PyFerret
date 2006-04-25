      SUBROUTINE RANGEL(UAMIN,UAMAX,UMINR,UMAXR)
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
C**    @(#)rangel.f	1.1    3/10/88
C** V600:  1/06 *acm*  prevent values of LOG10(0.0)
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
C     CALL RANGEL(UMIN,UMAX,UMINR,UMAXR)
C
C     THIS SUBROUTINE FINDS TWO VALUES FOR A VARIABLE SUCH THAT
C     EACH IS AN INTEGER POWER OF TEN AND THE RANGE BETWEEN THEM
C     CONTAINS THE MINIMUM AND MAXIMUM VALUES OF THE VARIABLE.
C
C     THE VARIOUS PARAMETERS AND THEIR DEFINITIONS ARE LISTED
C     BELOW:
C        UMIN   - MINIMUM VALUE OF THE VARIABLE.
C        UMAX   - MAXIMUM VALUE OF THE VARIABLE.
C        UMINR  - MINIMUM VALUE OF THE RANGE.
C        UMAXR  - MAXIMUM VALUE OF THE RANGE.
C
C
C     TRANSFER THE PARAMETERS TO NEW VARIABLES.
C
      UMIN = ABS(UAMIN)
      UMAX = ABS(UAMAX)

C
C     CHECK THE PARAMETERS TO MAKE SURE THAT THEY ARE VALID.
C
      IF(UMIN .GT. UMAX)THEN
        TEMP = UMIN
        UMIN = UMAX
        UMAX = TEMP
      ENDIF
C
C     CALCULATE THE MINIMUM VALUE OF THE RANGE.
C
      IF (UMIN .EQ. 0.) THEN
         FM = 0.
      ELSE
         FM = ALOG10(UMIN)
      ENDIF
      M = IFIX(FM)
      IF(FM .LT. 0.0)M = M - 1
      UMINR = 10.0 ** M
C
C     CALCULATE THE MAXIMUM VALUE OF THE RANGE.
C
      IF (UMAX .EQ. 0.) THEN
         FM = 0.
      ELSE
         FM = ALOG10(UMAX)
      ENDIF
      M = IFIX(FM)
      IF(FM .GT. 0.0)M = M + 1
      UMAXR = 10.0 ** M
      RETURN
C
      END
