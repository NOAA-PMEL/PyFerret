	SUBROUTINE SHADE_SUB (data,xc,yc,xs,ys, pixel)

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
*       *acm* 11/02     V5.50 For rectangular shade plots with no pattern
*                       calls, use cellarray call for the shade.
*       *acm*  1/03     v5.50 Remove debug line that stopped Ferret for large
*                             cellarray plots (!)
*       *acm* 1/27/03   Bug fix in cellarray plots, which had not taken the
*                       possible /HLIMITS and /VLIMITS into account.
* v5.51*acm*  1/30/03  Comment out the cellarray implementation; xgks doesnt
*                      completely implement it & has a precision bug for 
*                      large plots.  see comment lines: c no_cellarray

	INCLUDE		'parampl5_dat.decl'
	INCLUDE		'PARAMPL5.DAT'
        INCLUDE         'shade_vars.cmn'
	INCLUDE		'axis_inc.decl'
	INCLUDE		'AXIS.INC'
	INCLUDE		'miss_inc.decl'
	INCLUDE		'MISS.INC'
        INCLUDE         'hd_inc.decl'
        INCLUDE         'HD.INC'
	INCLUDE		'cont_inc.decl'
	INCLUDE		'CONT.INC'
	INCLUDE		'gkscm1_inc.decl'
	INCLUDE		'GKSCM1.INC'
	INCLUDE		'gkscm2.cmn'
        INCLUDE         'pen_inc.decl'
        INCLUDE         'PEN.INC'
	INCLUDE         'gkspar.inc'
	INCLUDE 	'xinterrupt.cmn'	! 1/89 *sh*
	INCLUDE 	'ppl_in_ferret.cmn'
        INCLUDE         'switch_inc.decl'
        INCLUDE         'SWITCH.INC'

	INTEGER		error,i,j,trans_no,cflag

        INTEGER         my_trans,fill_ndx,next_ndx,GET_NDX
        DATA            my_trans/20/

C       data statement of my_trans added for linux port *jd* 12.96

	REAL		data(nx,ny),xc(nx),yc(ny),xs(nx+1),ys(ny+1)
	REAL	 	w(4),v(4),ndc(4),uc(4)
	REAL		xinv,yinv, cx1, cx2, cy1, cy2

	logical		valid

        INTEGER		pixel(*), pmax
	LOGICAL		ITS_BATCH_GRAPHICS, REGULARLY_SPACED,
     .                  batch, irregular, use_cell

* declarations for curvilinear coordinates
	LOGICAL		ITS_CURVILINEAR, ITS_CURV_EDGES, BAD_XY_COORD,
     .			curvilinear
	INTEGER		i1, npt, pt1
	INTEGER		maxpts, maxpt_ov2
	PARAMETER     ( maxpts = 200,
     .			maxpt_ov2 = maxpts / 2 )
	REAL		px(maxpts), py(maxpts)

*	PARAMETER	(my_trans = 10)
	EXTERNAL	GET_NDX

*	MY_TRANS is the normalization transformation used in laying down the
*	area fill overlay.

******************************************************************************

*       *sh* (3/97) are we doing a curvilinear plot?
	IF (ppl_in_ferret) THEN
	   curvilinear = ITS_CURVILINEAR()
	ELSE
	   curvilinear = .FALSE.
	ENDIF

*	*jd* mod to allow shading single level fields 
	IF (nlev2 .LT. 1) GOTO 1000
	shd_levels = max (nlev2 - 1,1)

* are we in no-X mode?
	batch = ITS_BATCH_GRAPHICS()

*	DETERMINE PATTERNS OR COLORS TO SET BUNDLED FILL AREA REPRESENTATIONS
	CALL set_fill_area_rep (wsid,wstype)

        IF (meta_actv .AND. .NOT. meta_clsplt) 
     .     CALL set_fill_area_rep (meta_wsid,meta_wstype)

*	GET PROPER SCALING, NDC COORDINATES, ETC FOR FILL
	CALL gqcntn (error,trans_no)
	IF (error .NE. 0) GOTO 1000

	CALL gqnt (trans_no,error,w,v)
	IF (error .NE. 0) GOTO 1000

	CALL get_user_coords (w,v,ndc,uc,xinv,yinv)

*	NOW SET THE CLIPPING RECTANGLE AROUND THEM AND PATTERN SIZE

	CALL gswn (my_trans,uc(1)*xinv,uc(2)*xinv,uc(3)*yinv,uc(4)*yinv)
	CALL gsvp (my_trans,ndc(1),ndc(2),ndc(3),ndc(4))
	CALL gselnt (my_trans)

	CALL gqclip (error,cflag,ndc)
	IF (error .NE. 0) GOTO 1000
	IF (cflag .NE. gclip) CALL gsclip (gclip)

*	SET UP COORDINATES FOR CENTER OF EACH GRID BOX

        irregular = .FALSE.

**** X coordinates ***
	IF (ppl_in_ferret) THEN			!	JD 1.2.90	

* "curvilinear" indicates a 3-argument SHADE command: "SHADE vals, xpos, ypos"
	   IF ( curvilinear ) THEN
              irregular = .FALSE.
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
	      CALL box_bounds ('X',xc(1),nx,xs,valid)
	      IF (.NOT. valid) THEN
	         interrupted = .true.
	         GOTO 3333
	      ENDIF
	   ENDIF
	   DO 20 i = 1,nx+1
	      xs(i) = xs(i)*xinv
20	   CONTINUE
	ELSE
	   xs(1) = (xc(1) - (xc(2) - xc(1))/2.0)*xinv
	   DO 30 i = 1,nx-1
	      xs(i+1) = (xc(i) + (xc(i+1) - xc(i))/2.0)*xinv
30	   CONTINUE
	   xs(nx+1) = (xc(nx) + (xc(nx) - xc(nx-1))/2.0)*xinv
	ENDIF

        IF (.NOT.curvilinear .AND. .NOT.irregular) 
     .      irregular = .NOT. REGULARLY_SPACED(xc, nx)

**** Y coordinates *** (see doc of logic in X coordinates)
	IF (ppl_in_ferret) THEN			!	JD 1.2.90
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
	      CALL box_bounds ('Y',yc(1),ny,ys,valid)
	      IF (.NOT. valid) THEN
	         interrupted = .true.
	         GOTO 3333
	      ENDIF
	      DO 31 j = 1,ny+1
	         ys(j) = ys(j)*yinv
31	      CONTINUE
	   ENDIF
	ELSE
	   ys(1) = (yc(1) - (yc(2) - yc(1))/2.0)*yinv
	   DO 35 j = 1,ny-1
	      ys(j+1) = (yc(j) + (yc(j+1) - yc(j))/2.0)*yinv
35	   CONTINUE
	   ys(ny+1) = (yc(ny) + (yc(ny) - yc(ny-1))/2.0)*yinv
*								END 12.13.89
	ENDIF
        IF (.NOT.curvilinear .AND. .NOT.irregular) 
     .      irregular = .NOT. REGULARLY_SPACED(yc, ny)


*       WRITE OUT SHD_LEVELS & SHADE SETTINGS TO METAFILE (4.91)
#ifndef xgks
        IF (meta_actv) CALL meta_shade_set
#endif
c no_cellarray
c
c* For cellarray shade plots:
c
c*       For version 5.5 cellarray called only for X output, when
c*       there is no metafile open, and we are not in -batch mode.
c*       cellarray call is for rectangular, evenly spaced grids,
c*       to be filled with solid colors.
c*       Use older calls using fillarea when there is meta or ps output.
c
c      use_cell = ((.NOT. meta_actv ) .AND. (.NOT.batch) .AND.
c     .            (.NOT.curvilinear) .AND. (mnpat.EQ.0) .AND.
c     .            (.NOT.irregular)   .AND. (.NOT.SOVER) ) 
c
c      IF (use_cell) THEN
c
c* Collect color indices in cellarray pixel array.
c
c        cx1 = uc(1)*xinv   
c        cx2 = uc(2)*xinv   
c        cy1 = uc(3)*yinv
c        cy2 = uc(4)*yinv
c
c* Get the number of grid cells to be plotted 
c
c        mx = 0
c        my = 0
c
c        DO 60 i = 1, nx
c           IF (xc(i) .GE. cx1 .AND. xc(i) .LE. cx2) mx = mx + 1
c  60    CONTINUE
c        DO 65 j = 1, ny
c           IF (yc(j) .GE. cy1 .AND. yc(j) .LE. cy2) my = my + 1
c  65    CONTINUE
c
c        k = 0
c        DO 80 j = 1,ny
c	   DO 70 i = 1,nx
c
c           IF ( interrupted ) GOTO 3333			! 1/89 - *sh*
c
c           IF (xc(i) .GT. cx1 .AND. xc(i) .LE. cx2  .AND.
c     .         yc(j) .GT. cy1 .AND. yc(j) .LE. cy2 ) THEN
c
c              fill_ndx  = GET_NDX (data(i,j),fill_ndx)
c              k = k + 1
c
c	      IF (fill_ndx .NE. -999) THEN
c                 pixel(k) = fill_ndx + lo_color_ndx(wsid)-2
c                 pmax = max (pixel(k), pmax)
c
c              ELSE
c                 pixel(k) = 0
c
c              ENDIF
c
c           ENDIF
c
c70         CONTINUE
c80      CONTINUE
c
c*  Render the field using the cellarray primitive call.
c
c         DO 90 i = 0, pmax
c            CALL gsfaci (i)
c90       CONTINUE
c
c         CALL gca (cx1, cy1, cx2, cy2, mx,my, 1,1, mx,my, pixel)
c      ELSE  ! non-cellarray:
c
c* Previous code for SHADE plot, when either a pattern is used or 
c* we have curvilinear coordinates

c no_cellarray


*	LAY FILL DOWN ON SCREEN IN PROPER PLACE
	DO 50 j = 1,ny

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
	   IF (i .EQ. nx + 1) GOTO 50

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
	      IF (i .EQ. nx + 1) GOTO 48
	      IF ( curvilinear ) THEN
	         npt = i + 1
	         IF ( npt .GE. maxpt_ov2 ) GOTO 48
	      ENDIF

              next_ndx = GET_NDX (data(i,j),next_ndx)
	      IF (.NOT.curvilinear) THEN
	         IF (BAD_XY_COORD(i,j,xs,py,fill_ndx)) GOTO 48
	      ENDIF
	      IF (fill_ndx .EQ. next_ndx) GOTO 45
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

	   CALL set_fill_ndx (fill_ndx)
	   CALL gfa (npt,px,py)

*	   TO RETAIN ABILITY FOR COLOR OR HATCHING IN METAFILE, WRITE 
*	   FILL_NDX TO PRIVATE ITEM -- no, forget it *jd* 10.22.92
*	   IF (meta_actv .AND. .NOT. area_bundles) 
*     .				CALL meta_fill_ndx (fill_ndx)

	   GOTO 40
50	CONTINUE

c no_cellarray
c      ENDIF
c no_cellarray


 3333	CONTINUE					! 1/89 - *sh*
* post-interrupt re-entry				! 1/89 - *sh*

*	BACK TO PREVIOUS CLIPPING
	CALL gsclip (cflag)

*	BACK TO OLD TRANSFORM
	CALL gselnt (trans_no)

*	DO THE KEY
	IF (do_key .AND. .NOT.interrupted) CALL key	! 1/89 - *sh*

	return
*	FATAL ERROR
1000	return
	end
