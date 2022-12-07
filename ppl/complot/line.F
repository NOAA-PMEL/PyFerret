      SUBROUTINE LINE(X,Y,IMARK,NAPTS)
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
C**    @(#)line.f	1.1    3/10/88
C**
C**

C***********************************************************************
C**
C**                 PLOT+ Scientific Graphics System
C**
C***********************************************************************
C**
C**
* 10/2022 *acm* Code cleanup: Declare arguments X and Y with (*), update DO loop

C
C     CALL LINE(X,Y,IMARK,NPTS)
C
C     THIS SUBROUTINE PLOTS A SERIES OF POINTS USING DATA SUP-
C     PLIED IN ARRAYS.  THE CURRENT PLOTTING MODE IS USED TO
C     DETERMINE THE MANNER IN WHICH THE POINTS ARE CONNECTED.
C
C     THE VARIOUS PARAMETERS AND THEIR DEFINITIONS ARE LISTED
C     BELOW:
C        X,Y    - ARRAYS CONTAINING THE COORDINATES OF THE
C                 POINTS TO BE PLOTTED IN USER'S UNITS.
C        IMARK  - DATA MARK CODE (0-88).
C        NPTS   - NUMBER OF POINTS TO BE PLOTTED.
C
C
      DIMENSION X(*),Y(*)
C
      IF(NAPTS .EQ. 0)RETURN
      NPTS = IABS(NAPTS)
C
C     MOVE TO THE FIRST POINT WITH THE PEN UP.
C
      CALL PLOT(X(1),Y(1),0,0)
      CALL MARK(IMARK)
      IF(NPTS .EQ. 1)RETURN
C
C     MOVE TO THE REST OF THE POINTS WITH THE PEN DOWN.
C
      DO I=2,NPTS
         CALL PLOT(X(I),Y(I),1,IMARK)
      ENDDO

      RETURN
C
      END
