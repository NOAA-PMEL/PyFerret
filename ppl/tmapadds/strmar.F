      subroutine strmar (xx1, yy1, xx2, yy2, vel, xf, yf) 

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
*  in any             or publicity to endorse or promote any products
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

*       *acm* 8/2001 separated from flowline_sub.F
*                    strmar came from NCAR streamline code.
*  *acm*  12/7/01 Trap cases with dx=dy=0; atan2 fcn returns NAN under OSF.
*                 Declare all variables.
*  *acm&   3/06   change the scale of arrow heads; values are actually the same.
* V68  *acm* 1/12  changes for double-precision ferret, single-precision pplus

      IMPLICIT NONE
      REAL a160
      PARAMETER (a160 = 160.0*3.1416/180.0)
      REAL xf, yf
      REAL arrlen, vel, dir, xx1, xx2, yy1, yy2, small, fac

        include 'vector_inc.decl'
        include 'VECTOR.INC'
        include 'xcurvilinear.cmn'

C     Curvilinear coordinates declarations
      INTEGER status
      real*8 x_curv, y_curv
      REAL addx1, addy1, addx2, addy2, xxc, yyc, xxd, yyd, xxe, yye

      DATA small/1.e-6/

*  If the points are too close together, return

      IF (ABS(xx2-xx1) .LT. small   .AND. 
     .    ABS(yy2-yy1) .LT. small) RETURN

*  Set directions for arrows.

      dir = atan2(yy2-yy1, xx2-xx1)

*  Set arrow length scale based on vel.
*  factor of 0.2 is for the fact that the arrow is 0.2 of 
*  the velocity arrows for regular vectors

      arrlen = vel/(xf+yf)
      IF (.NOT. curvilinear) arrlen = 0.2* arrlen

      addx1 = arrlen* cos(dir+a160)
      addy1 = arrlen* sin(dir+a160)
      addx2 = arrlen* cos(dir-a160)
      addy2 = arrlen* sin(dir-a160)

      IF (curvilinear) THEN

          x_curv = xx2
          y_curv = yy2
          CALL curv_coord (x_curv, y_curv, 1, 1.0, 1.0, status)
          xxc=x_curv
          yyc=y_curv
          CALL plot (xxc, yyc, 0, 0)

          x_curv = xx2 + addx1
          y_curv = yy2 + addy1
          CALL curv_coord (x_curv, y_curv, 1, 1.0, 1.0, status)
          xxd = x_curv
          yyd = y_curv

          CALL plot (xxd, yyd, 1, 0)
          CALL plot (xxc, yyc, 0, 0)

          x_curv = xx2 + addx2
          y_curv = yy2 + addy2
          CALL curv_coord (x_curv, y_curv, 1, 1.0, 1.0, status)
          xxe = x_curv
          yye = y_curv

          CALL plot (xxe, yye, 1, 0)
          CALL plot (xxc, yyc, 0, 0)

      ELSE

         CALL plot (xx2, yy2, 0, 0)
         CALL plot (xx2+addx1, yy2+addy1, 1, 0)
         CALL plot (xx2, yy2, 0, 0)
         CALL plot (xx2+addx2, yy2+addy2, 1, 0)
         CALL plot (xx2, yy2, 0, 0)

      ENDIF

      RETURN
      END
