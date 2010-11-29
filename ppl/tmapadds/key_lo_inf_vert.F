      SUBROUTINE key_lo_inf_vert (px, py, ndx, v_inc,
     .          annotate_key, lab_digits, height)
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
* Make the lowest box (triangle) for the color key when the low level is 
* value-to-minus-infinity
*
*  v581 *acm* 3/05 open levels (-INF) (INF)

      INCLUDE 	'parampl5_dat.decl'
      INCLUDE 	'PARAMPL5.DAT'
      INCLUDE   'cont_inc.decl'
      INCLUDE	'CONT.INC'           ! with cont_key
      INCLUDE	'axis_inc.decl' 
      INCLUDE	'AXIS.INC'           ! with xorg, yorg
      INCLUDE	'PLTL.INC'           ! with makep
      INCLUDE	'hd_inc.decl'
      INCLUDE	'HD.INC'             ! with zmin, zmax


* Calling arguments
      REAL px(*),py(*), h_inc, height
      INTEGER ndx, annotate_key, lab_digits

* Internal declerations
      INTEGER maxlen, slen, ldigits
      REAL x1, x2, x3, x4, y1, y2, y3, y4
      REAL delx, xx, yy, xxx, yyy
      CHARACTER*20 TM_FMT_DIGITS, text

* save px and py values
      y1 = py(1)
      y2 = py(2)
      y3 = py(3)
      y4 = py(4)

      x1 = px(1)
      x2 = px(2)
      x3 = px(3)
      x4 = px(4)

* triangle.

      delx = x3 - x1

      px(1) = px(1) + delx/2.
      px(2) = x3
      px(3) = x4

      py(1) = py(3) - v_inc
      py(2) = py(3)

*	   CALL FILL AREA FOR THIS COLOR/GRAY VALUE
       call set_fill_ndx (ndx)
       call gfa (3,px,py)

* Outline
       px(4) = px(1)
       py(4) = py(1)
       IF (.NOT. cont_key) call gpl (4,px,py)
 
* Annotate min
      IF (annotate_key .GT. 0) THEN
         maxlen = 20
         ldigits = lab_digits+1
         text = TM_FMT_DIGITS (zmin,ldigits,maxlen,slen)
	 xx = px(1)/1000.0 - xorg - (slen/2)* height
         yy = py(1)/1000.0 - yorg - 1.5* height
	 
* FILL AND SHADE USE DIFFERENT COORD SYSTEMS -- *jd* 6.4.93
         if (makep .eq. 'SHADE' .or. makep .eq. 'FILLPOL' .OR.
     .         makep .EQ. 'POLYGON') then
	    call point_convert (xx,yy,xxx,yyy)
         else if (makep .eq. 'FILL') then
	    xxx = xx
	    yyy = yy
         endif

         call symbel (xxx,yyy,0.,height,slen,text(1:slen))
      ENDIF

* Restore px, py

      py(1) = y1
      py(2) = y2
      py(3) = y3
      py(4) = y4

      px(1) = x1
      px(2) = x2
      px(3) = x3
      px(4) = x4

      RETURN
      END
