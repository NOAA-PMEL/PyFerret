	SUBROUTINE SHADE_SUB (data,xc,yc,xs,ys)

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
*
*       lots of places ---> PPLMOD_PPLCMD ---> PPLMOD_PLTIT
*               ---> PPLADD_SHADE ---> PPLADD_SHADE_SUB()
*
*	Uses GKS calls to display an image using fill area, with
*	a range of colors if available on the workstation, or grey scale 
*	(or other available) patterns if not.  FORTRAN binding is used.

*	J Davison JISAO/PMEL/TMAP 6.15.88

*	Modified to fit into PPL as the SHADE command routine
*	J Davison 8.12.88

*	*sh*  1/23/89 - added interrupt detection
*	*jd*  3.15.89 - modified to provide for metafile output and also
*			to support more workstations
*	*jd*  7.17.89 - to fix transform error
*	*jd*  8.28.89 - to certify clipping on for shade
*	*jd* 12.13.89 - to fill edge cells completely -- clipping prevents
*			fill area overflowing plot axes
*	*jd*   1.2.90 -	get box boundaries from Ferret if in Ferret; can have
*			irregular grids there
*	*jd*  5.16.90 - to speed up the shade execution, wait while repeated
*			cells of same color are added to list to fill.  Also
*			extracted part of code into 2 subroutines to clean up.
*       *jd*  4.11.91 - calls to write shd_levels & shade settings to metafile
*                       as private items and NOT set color rep or fill area
*                       rep explicitly in the metafile
*	*jd*  4.2.92  - Includes decision to NOT use area bundles if more
*			than 20 in ATC GKS
*	*jd*  7.31.92 - Mod to allow shading of single level fields
*	*jd*  10.1.92 - Accommodate protection of used colors
*	*jd*  10.12.92- Handle lack of sufficient fill area indices in ATC
*			GKS when using hatching. -- cancel this 10.22.92 
*	*jd*  10.26.930 Eliminate meta_shade_set but don't replace with
*				code to retain function, yet
*	*jd*  01.27.94  Fix bug in assigning shd_levels -- always >= 1
*	*sh*  03.07.97  Allow external transformation to curvilinear coords
*	*sh*  25.11.97  changed calling args of GET_NDX by eliminating "i,j"
*			and by returning -999 for no index found

#ifdef unix
	include		'tmap_pplv11inc/parampl5_dat.decl'
	include		'pplv11inc/PARAMPL5.DAT'
        include         'tmap_pplv11inc/shade_vars.cmn'
	include		'tmap_pplv11inc/axis_inc.decl'
	include		'pplv11inc/AXIS.INC'
	include		'tmap_pplv11inc/miss_inc.decl'
	include		'pplv11inc/MISS.INC'
        include         'tmap_pplv11inc/hd_inc.decl'
        include         'pplv11inc/HD.INC'
	include		'tmap_pplv11inc/cont_inc.decl'
	include		'pplv11inc/CONT.INC'
	include		'tmap_pplv11inc/gkscm1_inc.decl'
	include		'pplv11inc/GKSCM1.INC'
	include		'tmap_pplv11inc/gkscm2.cmn'
        include         'tmap_pplv11inc/pen_inc.decl'
        include         'pplv11inc/PEN.INC'

#else
	INCLUDE		'tmap_pplv11inc:tmap_parampl5.dat'
        INCLUDE         'tmap_pplv11inc:tmap_shade_vars.inc'
	INCLUDE		'tmap_pplv11inc:tmap_axis.inc'
	INCLUDE		'tmap_pplv11inc:tmap_miss.inc'
        INCLUDE         'tmap_pplv11inc:tmap_hd.inc'
	INCLUDE		'tmap_pplv11inc:tmap_cont.inc'
	INCLUDE		'tmap_pplv11inc:tmap_gkscm1.inc'
	INCLUDE		'tmap_pplv11inc:tmap_gkscm2.inc'
        INCLUDE         'tmap_pplv11inc:tmap_pen.inc'
#endif

#ifdef atc_gks
#   ifdef unix
	include        'atc_gksdir/gkspar.inc'
#   else
	INCLUDE        'atc_gksdir:gkspar.inc'
#   endif
#else
#   ifdef unix
  	include        'tmap_pplv11inc/gkspar.inc'
#   else
  	INCLUDE        'sys$library:gksdefs.bnd'
#   endif
#endif

#ifdef unix
	include 	'tmap_pplv11inc/xinterrupt.cmn'	! 1/89 *sh*
	include 	'tmap_pplv11inc/ppl_in_ferret.cmn'	
#else
	INCLUDE 	'TMAP_PPLV11INC:XINTERRUPT.CMN'	! 1/89 *sh*
	INCLUDE 	'TMAP_PPLV11INC:PPL_IN_FERRET.CMN'	
#endif

	integer		error,i,j,trans_no,cflag

        integer         my_trans,fill_ndx,next_ndx,GET_NDX
        data            my_trans/20/

C       data statement of my_trans added for linux port *jd* 12.96

	real		data(nx,ny),xc(nx),yc(ny),xs(nx+1),ys(ny+1)
	real	 	w(4),v(4),ndc(4),uc(4)
	real		xinv,yinv

	logical		valid

* declarations for curvilinear coordinates
	LOGICAL		ITS_CURVILINEAR, ITS_CURV_EDGES, BAD_XY_COORD,
     .			curvilinear
	INTEGER		i1, npt, pt1
	INTEGER		maxpts, maxpt_ov2
	PARAMETER     ( maxpts = 200,
     .			maxpt_ov2 = maxpts / 2 )
	REAL		px(maxpts), py(maxpts)

*	parameter	(my_trans = 10)
	external	GET_NDX

*	MY_TRANS is the normalization transformation used in laying down the
*	area fill overlay.

******************************************************************************

*       *sh* (3/97) are we doing a curvilinear plot?
	if (ppl_in_ferret) then
	   curvilinear = ITS_CURVILINEAR()
	ELSE
	   curvilinear = .FALSE.
	ENDIF

*	*jd* mod to allow shading single level fields 
	if (nlev2 .lt. 1) goto 1000
	shd_levels = max (nlev2 - 1,1)

*	DETERMINE PATTERNS OR COLORS TO SET BUNDLED FILL AREA REPRESENTATIONS
	call set_fill_area_rep (wsid,wstype)

*	GET PROPER SCALING, NDC COORDINATES, ETC FOR FILL
	call gqcntn (error,trans_no)
	if (error .ne. 0) goto 1000

	call gqnt (trans_no,error,w,v)
	if (error .ne. 0) goto 1000

	call get_user_coords (w,v,ndc,uc,xinv,yinv)

*	NOW SET THE CLIPPING RECTANGLE AROUND THEM AND PATTERN SIZE
	call gswn (my_trans,uc(1)*xinv,uc(2)*xinv,uc(3)*yinv,uc(4)*yinv)	
	call gsvp (my_trans,ndc(1),ndc(2),ndc(3),ndc(4))
	call gselnt (my_trans)

	call gqclip (error,cflag,ndc)
	if (error .ne. 0) goto 1000
	if (cflag .ne. gclip) call gsclip (gclip)

*	SET UP COORDINATES FOR CENTER OF EACH GRID BOX

**** X coordinates ***
	if (ppl_in_ferret) then			!	JD 1.2.90	

* "curvilinear" indicates a 3-argument SHADE command: "SHADE vals, xpos, ypos"
	   IF ( curvilinear ) THEN
* ... Indices have been passed to this routine instead of coordinates
*     The coordinates are inside Ferret - gotten through CURV_COORDS below 
	      IF ( ITS_CURV_EDGES(1) ) THEN
* ... User gave Ferret edge points as coords -- set indices to point to them
* 	e.g. 1, 2, 3, ..., N+1
	         DO 11 i = 1, nx
 11	         xs(i) = xc(i)
	         xs(nx+1) = xs(nx) + 1
	      ELSE
* ... User gave Ferret grid points as coords -- set indices to index midpoints
* 	e.g. 0.5, 1.5, ...,N+.5
	         xs(1) = (xc(1) - (xc(2) - xc(1))/2.0)
	         DO 12 i = 1,nx-1
	            xs(i+1) = (xc(i) + (xc(i+1) - xc(i))/2.0)
12	         CONTINUE
	         xs(nx+1) = (xc(nx) + (xc(nx) - xc(nx-1))/2.0)
	      ENDIF

* normal rectangular grid (non-curvilinear)
	   ELSE
	      call box_bounds ('X',xc(1),nx,xs,valid)
	      if (.not. valid) then
	         interrupted = .true.
	         goto 3333
	      end if
	   ENDIF
	   do 20 i = 1,nx+1
	      xs(i) = xs(i)*xinv
20	   continue
	else
	   xs(1) = (xc(1) - (xc(2) - xc(1))/2.0)*xinv
	   do 30 i = 1,nx-1
	      xs(i+1) = (xc(i) + (xc(i+1) - xc(i))/2.0)*xinv
30	   continue
	   xs(nx+1) = (xc(nx) + (xc(nx) - xc(nx-1))/2.0)*xinv
	end if

**** Y coordinates *** (see doc of logic in X coordinates)
	if (ppl_in_ferret) then			!	JD 1.2.90
	   IF ( curvilinear ) THEN
	      IF ( ITS_CURV_EDGES(2) ) THEN
	         DO 21 i = 1, ny
 21	         ys(i) = yc(i)
	         ys(ny+1) = ys(ny) + 1
	      ELSE
	         ys(1) = (yc(1) - (yc(2) - yc(1))/2.0)
	         DO 22 i = 1,ny-1
	            ys(i+1) = (yc(i) + (yc(i+1) - yc(i))/2.0)
22	         CONTINUE
	         ys(ny+1) = (yc(ny) + (yc(ny) - yc(ny-1))/2.0)
	      ENDIF
	   ELSE
	      call box_bounds ('Y',yc(1),ny,ys,valid)
	      if (.not. valid) then
	         interrupted = .true.
	         goto 3333
	      end if
	      do 31 j = 1,ny+1
	         ys(j) = ys(j)*yinv
31	      continue
	   ENDIF
	else
	   ys(1) = (yc(1) - (yc(2) - yc(1))/2.0)*yinv
	   do 35 j = 1,ny-1
	      ys(j+1) = (yc(j) + (yc(j+1) - yc(j))/2.0)*yinv
35	   continue
	   ys(ny+1) = (yc(ny) + (yc(ny) - yc(ny-1))/2.0)*yinv
*								END 12.13.89
	end if

*       WRITE OUT SHD_LEVELS & SHADE SETTINGS TO METAFILE (4.91)
#ifndef xgks
        if (meta_actv) call meta_shade_set
#endif
*	LAY FILL DOWN ON SCREEN IN PROPER PLACE
	do 50 j = 1,ny

! initialize these so that the check in BAD_XY_COORD doesn't get caught
! in a loop
           py(1) = ys(j)
           py(2) = ys(j+1)
           py(3) = ys(j+1)
           py(4) = ys(j)

* ... initialize for outer DO...WHILE loop
	   i = 0

* ... outer: DO WHILE i LE nx ...
40	   i = i + 1					! To line 50 JD 5.16.90
	   if (i .eq. nx + 1) goto 50

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
	   if (fill_ndx .eq. -999) goto 40
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
	      if (i .eq. nx + 1) goto 48
	      IF ( curvilinear ) THEN
	         npt = i + 1
	         IF ( npt .GE. maxpt_ov2 ) GOTO 48
	      ENDIF

              next_ndx = GET_NDX (data(i,j),next_ndx)
	      IF (.NOT.curvilinear) THEN
	         IF (BAD_XY_COORD(i,j,xs,py,fill_ndx)) GOTO 48
	      ENDIF
	      if (fill_ndx .eq. next_ndx) goto 45
* ... inner: end loop

* found last cell in chain - now assign the vertices of the polygon
48	   IF ( curvilinear ) THEN
* ... after transformation this will be a curvy "tube" of one color
	      npt = i - pt1 + 1
* ... polygon chain begins along upper edge of row of cells
	      DO 100 i1 = 1, npt
	         px(i1) = xs(pt1+i1-1)
	         py(i1) = ys(j+1)
 100	      CONTINUE
* ... and returns (closes) along lower edge of row of cells
	      DO 101 i1 = npt+1, 2*npt
	         px(i1) = px(2*npt+1-i1)
	         py(i1) = ys(j)
 101	      CONTINUE
	      npt = 2*npt
	      CALL CURV_COORD(px, py, npt, xinv, yinv, status) ! xform the pts
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

	   call set_fill_ndx (fill_ndx)
	   call gfa (npt,px,py)

*	   TO RETAIN ABILITY FOR COLOR OR HATCHING IN METAFILE, WRITE 
*	   FILL_NDX TO PRIVATE ITEM -- no, forget it *jd* 10.22.92
*	   if (meta_actv .and. .not. area_bundles) 
*     .				call meta_fill_ndx (fill_ndx)

	   goto 40
50	continue

 3333	CONTINUE					! 1/89 - *sh*
* post-interrupt re-entry				! 1/89 - *sh*

*	BACK TO PREVIOUS CLIPPING
	call gsclip (cflag)

*	BACK TO OLD TRANSFORM
	call gselnt (trans_no)

*	DO THE KEY
	if (do_key .AND. .NOT.interrupted) call key	! 1/89 - *sh*

	return
*	FATAL ERROR
1000	return
	end
