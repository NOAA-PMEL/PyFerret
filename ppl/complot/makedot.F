      SUBROUTINE makedot

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
C  Ansley Manke 8/2001
C  Plot a single point making polymark calls rather than polyline
C  based on XYZPLT
C  10/2001 simplifications that go with buffering the GPM calls
C  
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
C     Check the plotting mode.

      IF(PMODE .EQ. ALPHA)GO TO 30


      XNEW = XTEMP
      YNEW = YTEMP
      ZNEW = ZTEMP

  30  IF(.NOT. WINDOF)GO TO 40

C     Calculate the new off-window flag.

      OWFLG = 0
      IF(XNEW .GT. XMAX)OWFLG = 1
      IF(XNEW .LT. XMIN)OWFLG = 2
      IF(YNEW .GT. YMAX)OWFLG = OWFLG + 4
      IF(YNEW .LT. YMIN)OWFLG = OWFLG + 8

C     CHECK THE NEW POINT TO SEE IF IT IS WITHIN THE BOUNDARY OF
C     THE WINDOW.  IF NOT, DO NOT PLOT IT.

      IF(OWFLG .NE. 0)GO TO 50

C     Transform the point to the screen coordinate system.

  40  CALL TFORM(XNEW,YNEW,ZNEW)

C     Plot the point.

      CALL ABPLT_makedot

C     Move the coordinates of the new point to the old point.

  50  XOLD = XNEW
      YOLD = YNEW
      ZOLD = ZNEW
      LOWFLG = OWFLG
      SAVEF = .FALSE.

C     CALCULATE THE OLD OFF-SCREEN FLAG.

      LOSFLG = 0
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


  60  RETURN
      END


