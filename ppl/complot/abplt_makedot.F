      SUBROUTINE ABPLT_makedot
                
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

*  8/2001  acm  for PLOT/SYMBOL=DOT
*  Simplified version of ABPLT to be called by make_dot.  no need
*  to compute intercepts for points off-screen.  If off-screen do
*  not plot

****************************************************************
*     CALL ABPLT

*     THIS SUBROUTINE PERFORMS SCREEN CLIPPING AND THEN MOVES THE
*     PEN FROM THE CURRENT POSITION TO A NEW POINT.


#ifdef unix
      include 'pltcom_dat.decl'
      include 'PLTCOM.DAT'
#else
      INCLUDE 'tmap_pplv11inc:tmap_PLTCOM.DAT'
      INCLUDE 'pplv11inc:pltcom.dat'
#endif
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

C     CHECK THE NEW POINT TO SEE IF IT IS WITHIN THE BOUNDARY OF
C     THE SCREEN.  IF NOT, DO NOT PLOT IT.

  80  IF(OSFLG .NE. 0)GO TO 100

C     PLOT TO THE POINT WITH THE PEN UP AND THEN DRAW A DOT.

      A = ANEW - AMIN
      B = BNEW - BMIN
      PEN = .FALSE.
      CALL ABMV_makedot
      PEN = .TRUE.
      CALL ABMV_makedot
      GO TO 100

C     CHECK THE NEW POINT TO SEE IF IT IS WITHIN THE BOUNDARY OF
C     THE SCREEN.  IF NOT, DO NOT PLOT IT.

  90  IF(OSFLG .NE. 0)GO TO 100

C     PLOT TO THE POINT WITH THE PEN UP.

      A = ANEW - AMIN
      B = BNEW - BMIN
      PEN = .FALSE.
      CALL ABMV_makedot

C     MOVE THE COORDINATES OF THE NEW POINT TO THE OLD POINT.

 100  AOLD = ANEW
      BOLD = BNEW
      AOSOLD = A
      BOSOLD = B
      LOSFLG = OSFLG
      RETURN

      END
