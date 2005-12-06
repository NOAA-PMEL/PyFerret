      SUBROUTINE SHADE_FILL_IT (data, nxdim, nydim, px, py, xs, ys, 
     .                          xinv, yinv, xadd, lmin, lmax, 
     .                          ilo, ihi, jlo, jhi, status)

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
* 2/23/2005 acm
* Do the fill for a shade plot, perhaps with a modulo offset in X

       INCLUDE 'parampl5_dat.decl'
       INCLUDE 'PARAMPL5.DAT'
       INCLUDE 'hd_inc.decl'
       INCLUDE 'HD.INC'
       INCLUDE 'ferret.parm'
       INCLUDE 	'xinterrupt.cmn'	! 1/89 *sh*
       INCLUDE 'errmsg.parm'
       INCLUDE 'xcurvilinear.cmn'

       INTEGER nxdim, nydim, status
       REAL    data(nxdim,nydim), px(*), py(*), xs(*), ys(*), lmin, lmax

* local definitions

      REAL    minrowx, maxrowx, big
      INTEGER GET_NDX, i, j, i1, fill_ndx, npt, pt1, next_ndx,
     .        jtop
      LOGICAL BAD_XY_COORD
      CHARACTER*128 buff

      big = ABS(bad_val4)

      status = ferr_ok
      lmin = -1*big
      lmax = big

*  check for consistency: data range 1:ny, index range jlo:jhi  
*  If inconsistent, use whole Y range given by ny

      IF (curvilinear) THEN
         IF (jlo .LT. 1 .OR. jhi .GT. ny+1) THEN  
            jlo = 1
            jtop = ny
         ELSE
            jtop = jhi 
            IF (jhi .EQ. ny+1) jtop = ny
         ENDIF
      ELSE
         jlo = 1
         jtop = ny
      ENDIF

*  Likewise for X, if this is not an x modulo plot

      IF (curvilinear .AND. (.NOT. xmodulo) ) THEN
         IF (ilo .LT. 1 .OR. ihi .GT. nx+1) THEN  
            ilo = 1
            itop = nx
         ELSE
            itop = ihi 
            IF (ihi .EQ. nx+1) itop = nx
         ENDIF
      ELSE
         ilo = 1
         itop = nx
      ENDIF

*	LAY FILL DOWN ON SCREEN IN PROPER PLACE

	DO 50 j = jlo, jtop

! initialize these so that the check in BAD_XY_COORD doesn't get caught
! in a loop
           py(1) = ys(j)
           py(2) = ys(j+1)
           py(3) = ys(j+1)
           py(4) = ys(j)

* ... initialize for outer DO...WHILE loop
	   i = 0

           minrowx = big
           maxrowx = -1.*big

* ... outer: DO WHILE i LE itop ...
40	   i = i + 1					! To line 50 JD 5.16.90
	   IF (i .EQ. itop+1) GOTO 49

	   IF ( interrupted ) GOTO 3333			! 1/89 - *sh*

* ... initialize for inner DO...WHILE loop
	   IF (.NOT.curvilinear) THEN
* ... note that for curvilinear plots the screening of coordinates is NOT
*	applied. This represents a minor bug/feature since the user might
*	expect the PLOT+ limits command still to apply.
*	(It was just too much trouble for the potential benefit *sh*)
	      IF (BAD_XY_COORD(i,j,xs,py,fill_ndx)) GOTO 40
	   ENDIF
	   fill_ndx = GET_NDX (data(i,j),fill_ndx)
	   IF (fill_ndx .EQ. -999) GOTO 40
	   pt1 = i
	   npt = 1

* ... inner: DO WHILE fill_ndx remains unchanged ...
* ... find the largest box from this row that is all this color (optimization)
* ... Note: further optimization is possible (especially for very noisy fields)
*	    for curvilinear plots. Since each polygon "tube" shares 2 vertices
*	    with the one that preceded it on the row, we are unnecessarily
*	    computing the curvilinear position of these points TWICE in calls
*	    to CURV_COORD
45	      i = i + 1
	      IF (i .EQ. itop+1) GOTO 48
	      IF ( curvilinear ) THEN
	         npt = i + 1
	         IF ( npt .GE. maxpt_ov2 ) GOTO 48
	      ENDIF

              next_ndx = GET_NDX (data(i,j),next_ndx)
	      IF (.NOT.curvilinear) THEN
	         IF (BAD_XY_COORD(i,j,xs,py,fill_ndx)) GOTO 48
	      ENDIF
              IF (curvilinear) GOTO 48  ! each cell on its own.
	      IF (fill_ndx .EQ. next_ndx) GOTO 45
* ... inner: end loop

* found last cell in chain - now assign the vertices of the polygon
48	   IF ( curvilinear ) THEN
* ... after transformation this will be a curvy "tube" of one color
	      npt = i - pt1 + 1
* ... polygon chain begins along upper edge of row of cells
	      DO 51 i1 = 1, npt
	         px(i1) = xs(pt1+i1-1)
	         py(i1) = ys(j+1)
 51	      CONTINUE
* ... and returns (closes) along lower edge of row of cells
	      DO 52 i1 = npt+1, 2*npt
	         px(i1) = px(2*npt+1-i1)
	         py(i1) = ys(j)
 52	      CONTINUE
	      npt = 2*npt
	      CALL CURV_COORD_ADD (px, py, npt, xinv, yinv, xadd,
     .                        status) ! xform the pts
               
              minrowx = MIN( minrowx, px(1) )
              minrowx = MIN( minrowx, px(2) )
              maxrowx = MAX( maxrowx, px(1) )
              maxrowx = MAX( maxrowx, px(2) )

	      IF ( status .NE. 0 ) RETURN
	   ELSE
* ... always a rectangle -- just save the corners
	      px(1) = xs(pt1)
	      px(2) = xs(pt1)
	      px(3) = xs(i)
	      px(4) = xs(i)
	      py(1) = ys(j)
	      py(2) = ys(j+1)
	      py(3) = ys(j+1)
	      py(4) = ys(j)
	      npt = 4
	   ENDIF
	   pt1 = i	! index of left edge of first cell of next color
	   i = i - 1

	   CALL set_fill_ndx (fill_ndx)
	   CALL gfa (npt,px,py)

*	   TO RETAIN ABILITY FOR COLOR OR HATCHING IN METAFILE, WRITE 
*	   FILL_NDX TO PRIVATE ITEM -- no, forget it *jd* 10.22.92
*	   IF (meta_actv .AND. .NOT. area_bundles) 
*     .				CALL meta_fill_ndx (fill_ndx)

	   GOTO 40

* For each y, what is the max and min x coordinate value covered
* by all the rows and columns?
* We want to save the minimum of the high coords for each row,
* and the maximum of the low coords for each row 

49         IF (minrowx .LT. big) lmin = MAX(lmin, minrowx)
           IF (maxrowx .GT. -1.*big) lmax = MIN(lmax, maxrowx)

50	CONTINUE

        RETURN

3333    status = ferr_interrupt
        RETURN

* error exit
 5000	RETURN
	END 
