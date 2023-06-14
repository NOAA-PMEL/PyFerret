	SUBROUTINE SHADE_SUB (data,xc,yc,xs,ys, cells)

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
* V5.54 *acm* 2/11/04  For cellarray plots, if the data is all missing or
*                      outside the range of the axes that have been plotted,
*                      do not make the cellarray call.
*  V5.54 *acm* 2/11/04 For cellarray plots, make sure the edges of the rectangle 
*                      of data corresponds to the correct part of the box drawn
*                      by the axes. (/hlim or /vlim plots)
* V5.61 *acm* 4/16/04  For cellarray plots, fix treatment of half-cells around
*                      the edges of the plot.  
* V5.81 *acm* 3/05     For curvilinear plots, do a partial solution for the question
*                      of modulo longitude axes: Determine the index range needed
*                      to plot the requested /HLIM/VLIM range. New routine SHADE_FILL_IT
*                      uses only that range, checks to see if a branch point has been 
*                      crossed, and makes the plot call.  For curvilinear plots, do not 
*                      gather up whole groups of cells of the same color but do each 
*                      separately, checking for the branch cut.  Make more calls to 
*                      SHADE_FILL_IT as needed to fill the longitude range.
C V602 *acm* 2/07 check status coming out of CURV_COORD_RANGE; part of fix to bug 1485
* V603 *acm* 5/07 Fix part 1 of bug 1302; if xmodulo is true, then set dim1_modulo true.
* V6.8 *acm* 7/9/2012 For batch graphics on RHEL6, change ITS_BATCH_GRAPHICS
*                     from logical to integer, then batch = ITS_BATCH_GRAPHICS().NE.0
* V701 *acm* 7/17 ticket 2451: cellarray plots with axis direction reversed by /VLIM or /HLIM

        IMPLICIT NONE
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
        INCLUDE         'xcurvilinear.cmn'
        INCLUDE         'errmsg.parm'

	INTEGER		error,i,j,k,trans_no,cflag

        INTEGER         my_trans,fill_ndx,GET_NDX
        DATA            my_trans/20/

C       data statement of my_trans added for linux port *jd* 12.96

	REAL		data(nx,ny),xc(nx),yc(ny),xs(nx+1),ys(ny+1)
	REAL	 	w(4),v(4),ndc(4),uc(4)
	REAL		xinv,yinv, cx1, cx2, cy1, cy2, half

	logical		valid

        INTEGER		cells(*), pmax
        INTEGER         j1, jn, jd, my, mx
	LOGICAL		REGULARLY_SPACED,
     .                  irregular, plot_cell, do_cell

* declarations for curvilinear coordinates
	LOGICAL		ITS_CURVILINEAR, ITS_CURV_EDGES
	INTEGER		status
	INTEGER		maxpts, maxpt_ov2, nright, nleft, 
     .                  ilo, ihi, jlo, jhi
	PARAMETER     ( maxpts = 200,
     .			maxpt_ov2 = maxpts / 2 )
	REAL		px(maxpts), py(maxpts), xadd, xrange, lmin, lmax

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
	shd_levels = MAX (nlev2 - 1,1)

*	DETERMINE PATTERNS OR COLORS TO SET BUNDLED FILL AREA REPRESENTATIONS
	CALL set_fill_area_rep (wsid,wstype)

*	GET PROPER SCALING, NDC COORDINATES, ETC FOR FILL
	CALL FGD_GQCNTN(error,trans_no)
	IF (error .NE. 0) GOTO 1000

	CALL FGD_GQNT(trans_no,error,w,v)
	IF (error .NE. 0) GOTO 1000

	CALL get_user_coords (w,v,ndc,uc,xinv,yinv)

*	NOW SET THE CLIPPING RECTANGLE AROUND THEM AND PATTERN SIZE

	CALL FGD_GSWN(my_trans,uc(1)*xinv,uc(2)*xinv,uc(3)*yinv,uc(4)*yinv)
	CALL FGD_GSVP(my_trans,ndc(1),ndc(2),ndc(3),ndc(4))
	CALL FGD_GSELNT(my_trans)

	CALL FGD_GQCLIP(error,cflag,ndc)
	IF (error .NE. 0) GOTO 1000
	IF (cflag .NE. gclip) CALL FGD_GSCLIP(gclip)

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
	           xs(i) = xc(i)
11	         CONTINUE
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
 	           ys(i) = yc(i)
21	         CONTINUE
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
c
c* For cellarray shade plots:
c
c*       For version 5.52 cellarray called only for gif batch mode.
c*       cellarray call is for rectangular, evenly spaced grids,
c*       to be filled with solid colors. Use previously-implemented 
c*       calls tha use fillarea when there is meta or ps output.
c*       use_cell is determined in disp_set_up, called on the Ferret 
c*       side from xeq_shade.

* See ticket 2039:  When the cellarray plot is one of several in viewports,
* the colors in the cellarray image are off by one. Have not found the cause. 
* Work around it by refusing to make a cellarray call except for the first 
* color map, either the first plot on the page, or if a PPL SHASET RESET has 
* been issued.

* For ticket 2451: When the axes are reversed, the cellarray call draws nothing.
* Also refuse to make the call in this case.

      do_cell = use_cell .AND. (lo_color_ndx(wsid).EQ.num_line_colors+1) .AND. 
     .    xinv.EQ.1 .AND. yinv.EQ.1

      IF (do_cell) THEN
      
* Collect color indices in cellarray cells array.
* Set the data rectangle to represent whole grid cells.

        half = (xc(2) - xc(1)) / 2.
        cx1 = (uc(1) - half)* xinv   
        cx2 = (uc(2) + half)* xinv   

        half = (yc(2) - yc(1)) / 2.
        cy1 = (uc(3) - half)* yinv
        cy2 = (uc(4) + half)* yinv

* Get the number of grid cells to be plotted 

        mx = 0
        my = 0

        plot_cell = .TRUE.
        DO 60 i = 1, nx
           IF (xc(i) .GE. cx1 .AND. xc(i) .LE. cx2) mx = mx + 1
60	CONTINUE
        DO 65 j = 1, ny
           IF (yc(j) .GE. cy1 .AND. yc(j) .LE. cy2) my = my + 1
65	CONTINUE

* none of the data is in the range of the axes we are plotting; 
* dont make the cellarray call.

        IF ( (mx .EQ. 0) .OR. (my .EQ. 0) ) plot_cell = .FALSE.

* load the color array

        k = 0
        pmax = 0
        j1 = ny
        jn = 1
        jd = -1

        IF (yinv .LT. 0) THEN
           j1 = 1
           jn = ny
           jd = 1
        ENDIF

        DO 80 j = j1,jn,jd
	   DO 70 i = 1,nx

           IF ( interrupted ) GOTO 3333			! 1/89 - *sh*

           IF (xc(i) .GT. cx1 .AND. xc(i) .LE. cx2  .AND.
     .         yc(j) .GT. cy1 .AND. yc(j) .LE. cy2 ) THEN

              fill_ndx  = GET_NDX (data(i,j),fill_ndx)
              k = k + 1
 
              IF (fill_ndx .NE. -999) THEN
                 cells(k) = fill_ndx + lo_color_ndx(wsid)-2
                 pmax = max (cells(k), pmax)

              ELSE
                 cells(k) = 0
              ENDIF

           ENDIF

70	  CONTINUE
80	CONTINUE

c If all the data is missing don't make the cellarray call

        IF (pmax .EQ. 0) plot_cell = .FALSE.

*  Render the field using the cellarray primitive call.  

         IF (plot_cell) THEN
            DO 90 i = 0, pmax
               CALL FGD_GSFACI(i)
90          CONTINUE

* If the data rectangle doesnt take up the whole plot box, reset
* the world coordinates of the corners to match the data rectangle.

            half = (xc(2) - xc(1)) / 2.
            IF (cx1 .LT. xc(1) - half ) cx1 = xc(1)  - half
            IF (cx2 .GT. xc(nx) + half) cx2 = xc(nx) + half

            half = (yc(2) - yc(1)) / 2.
            IF (cy1 .LT. yc(1) - half ) cy1 = yc(1)  - half
            IF (cy2 .GT. yc(ny) + half) cy2 = yc(ny) + half
            IF (yinv .LT. 0) THEN
               half = -1*cy1  ! temp. storage to switch values
               cy1 = -1*cy2
               cy2 = half
            ENDIF
            CALL FGD_GCA(cx1, cy1, cx2, cy2, mx,my, 1,1, mx,my, cells)
         ENDIF

      ELSE  ! non-cellarray:

* SHADE plot, when either a pattern is used or 
* we have curvilinear coordinates

*	LAY FILL DOWN ON SCREEN IN PROPER PLACE

* Get range of indices for the curvi X,Y coordinates actually needed for the 
* hlimits, vlimits that were requested.

        IF (curvilinear) THEN
           CALL CURV_COORD_RANGE (uc, ilo, ihi, jlo, jhi, status )
           IF (status .NE. ferr_ok) GOTO 1000
           IF (xmodulo) dim1_modulo = .TRUE.
        ENDIF

        xadd = 0.
        CALL SHADE_FILL_IT (data, nx, ny, px, py, xs, ys, xinv, yinv, 
     .                   xadd, lmin, lmax, ilo, ihi, jlo, jhi, 
     .                   maxpt_ov2, status)
        IF (status .EQ. ferr_interrupt) GOTO 3333

*	MODULO REPLICATIONS TO FILL OUT uc(1) TO uc(2)

        IF (xmodulo .AND. curvilinear) THEN
           xrange = 360
           IF (.NOT. units_degrees) xrange = wwxmax - wwxmin
           nleft = 0
           nright = 0
           IF (lmin .GT. uc(1) ) nleft =  1 + INT(lmin - uc(1) )/ xrange
           IF (lmax .LT. uc(2) ) nright = 1 + INT(uc(2) - lmax )/ xrange

           IF (nleft .GT. 0) THEN
              xadd = 0.
              DO 100 i = 1, nleft
                 xadd = xadd - xrange
                 CALL SHADE_FILL_IT (data, nx, ny, px, py, xs, ys,
     .                         xinv, yinv,  xadd, lmin, lmax, ilo, 
     .                         ihi, jlo, jhi, maxpt_ov2, status)
                 IF (status .EQ. ferr_interrupt) GOTO 3333
100	      CONTINUE
           ENDIF

           IF (nright .GT. 0) THEN
              xadd = 0.
              DO 110 i = 1, nright
                 xadd = xadd + xrange
                 CALL SHADE_FILL_IT (data, nx, ny, px, py, xs, ys,
     .                         xinv, yinv,  xadd, lmin, lmax, ilo, 
     .                         ihi, jlo, jhi, maxpt_ov2, status)
                 IF (status .EQ. ferr_interrupt) GOTO 3333
110	      CONTINUE
           ENDIF


        ENDIF   ! xmodulo

      ENDIF	! use_cell

 3333	CONTINUE					! 1/89 - *sh*
* post-interrupt re-entry				! 1/89 - *sh*

*	BACK TO PREVIOUS CLIPPING
	CALL FGD_GSCLIP(cflag)

*	BACK TO OLD TRANSFORM
	CALL FGD_GSELNT(trans_no)

*	DO THE KEY
	IF (do_key .AND. .NOT.interrupted) CALL key	! 1/89 - *sh*
        CALL shade_key_cont(0)

	RETURN
*	Fatal error
1000	RETURN
	END
