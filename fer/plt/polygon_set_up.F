	SUBROUTINE POLYGON_SET_UP
     .			      ( memory,
     .				overlay, transpz, coord_ax, no_labels,
     .				no_range, only_val,
     .				symbol, line_type,
     .				mr_list, cx_list, nmv,
     .				x_dat, y_dat, status )


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

* Set up for single or multiple polygons to be  ploted by loading data
* and labels and defining axis formats

* The POLYGON command accepts single and multi-dimensional arguments
* "1D FORM"
*	   yes? POLYGON xpoly1D, ypoly1D, values
*   where if xpoly1D or ypoly1D contain missing values, those represent
*   the end of one polygon and the start of the next. The length of the values
*   array must equal the number of polygons
* in which case the X coordinate might ve visualized as
*    x1,x1,x1,x1,BAD,x2,x2,x2,BAD,x3,x3,x3,x3,x3,x3,x3,x3,BAD,...
* where the "1","2","3" refer to the successive polygons

* "2D FORM"
*	   yes?  POLYGON xpoly2D, ypoly2D, values
*   where values must be 1-dimensional and its axis must match in size and
*   orientation one of the axes from the 2D arrays. The other axis of the
*   2D coordinates is the axis of the coordinates within each polygon
* in which case the X coordinate might ve visualized as
*    x1,x1,x1,x1,BAD,BAD,...
*    x2,x2,x2,BAD,BAD,...
*    x3,x3,x3,x3,x3,x3,x3,x3,BAD,BAD,...

* programmer - steve hankin
* NOAA/PMEL, Seattle, WA - Tropical Modeling and Analysis Program

* V500: 2/99 - based on plot_set_up.F
*       3/99 - *jd* mods to complete the command
* V500: 3/99 *kob* - upped VAR_UNITS from 32 to 64 chars
*       5/99 *kob* - minor mod to remove improper comment line linux compiler
*                    didn't like
*       6/99 - *jd* fix bug in 2 var form where x_dat(1) is reassigned

#ifdef unix
	include 'tmap_dims.parm'
	include 'xtm_grid.cmn_text'
	external xgt_grid_data
	include 'ferret.parm'
	include 'plot_setup.parm'
	include 'errmsg.parm'
	include 'slash.parm'
!	include 'gfdl_vms.parm'
	include 'xplot_setup.cmn'
	include 'xcontext.cmn'
	include 'xvariables.cmn'
	include 'xprog_state.cmn'
	include 'xplot_state.cmn'
	include 'xtext_info.cmn'
	include 'axis_inc.decl'       ! with axis lengths
	include 'AXIS.INC'       ! with axis lengths	
	include 'ppldat_inc.decl'       ! with nsize
	include 'PPLDAT.INC'       ! with nsize
        include 'switch_inc.decl'
        include 'SWITCH.INC'
        include 'pltl_inc.decl'
        include 'PLTL.INC'
#else
	INCLUDE 'TMAP_FORMAT:TMAP_DIMS.PARM'
	INCLUDE 'TMAP_FORMAT:XTM_GRID.CMN'
	INCLUDE 'FERRET_CMN:FERRET.PARM'
	INCLUDE 'FERRET_CMN:PLOT_SETUP.PARM'
	INCLUDE 'FERRET_CMN:ERRMSG.PARM'
	INCLUDE 'FERRET_CMN:SLASH.PARM'
	INCLUDE 'FERRET_CMN:XPLOT_SETUP.CMN'
	INCLUDE 'FERRET_CMN:XCONTEXT.CMN'
	INCLUDE 'FERRET_CMN:XVARIABLES.CMN'
	INCLUDE 'FERRET_CMN:XPROG_STATE.CMN'
	INCLUDE 'FERRET_CMN:XTEXT_INFO.CMN'
	INCLUDE 'TMAP_PPLV11INC:AXIS_INC.DECL/LIST'       ! with axis lengths
	INCLUDE 'PPLV11INC:AXIS.INC/LIST'       ! with axis lengths	
	INCLUDE 'TMAP_PPLV11INC:PPLDAT_INC.DECL/LIST'       ! with nsize
	INCLUDE 'PPLV11INC:PPLDAT.INC/LIST'       ! with nsize
        INCLUDE 'tmap_pplv11inc:tmap_SWITCH.INC'
        INCLUDE 'TMAP_PPLV11INC:PLTL_INC.DECL'
        INCLUDE 'PPLV11INC:PLTL.INC'
#endif

* calling argument declarations:
	LOGICAL		overlay, transpz, no_labels, no_range
	INTEGER		symbol, line_type, coord_ax, iunits,
     .			nmv, mr_list( nmv ), cx_list( nmv ), status
	REAL		memory( mem_blk_size, max_mem_blks )
	REAL		x_dat(*), y_dat(*)     ! dynamic space allocation 
	REAL            dummy_x_dat, only_val

* internal variable declarations:
	LOGICAL         missing_polygon
	LOGICAL		NO_LINE_RANGE, ambiguous_coord_axis,
     .			formatted, user_title, oneD_form, stack_up
	CHARACTER*200	KEY_STRING, FULL_VAR_TITLE, plot_title
* 3/99 *kob* - upped VAR_UNITS from 32 to 64 chars
	CHARACTER	VAR_UNITS*64, LEFINT*4,
     .			x_ax*1, y_ax*1,buff*8, buff2*4, t1_date*14
	INTEGER         TM_UNIT_ID
	INTEGER		TM_LENSTR1, CX_DIM_LEN, CGRID_SIZE,
     .			mr_x, cx_x, mr_y, cx_y, cx_color, mr_color,
     .			slen, slen2, x_lab, y_lab, nline_in_mem, i,
     .			npolygon, ncolor, ipoly, xrow, yrow, start, end,
     .			npts, idim, color_dim, coord_dim, i0,
     .			style, justify, ndim, dims(4), nfdim, fdims(4),
     .			plot_mem_used, pxlim, pylim, ptitl,
     .			cx_buff_x, cx_buff_y
	REAL		x_len, y_len, lo, hi, lox, hix, loy, hiy,
     .			xkeypos, ykeypos, xkeysize, ykeysize, lab_ht,
     .			dt_min
	REAL*8		vmin, vmax, delta

* signal that plot set-up has begun
	IF ( mode_diagnostic ) CALL SPLIT_LIST(pttmode_ops, ttout_lun,
     .				'setting up polygons', 19)

* Permit 2 arguments, ginn up dummy 3rd in that case
        IF (nmv .EQ. 3) THEN
	   cx_color = cx_list(3)
	   mr_color = mr_list(3)
        ELSEIF (nmv .EQ. 2) THEN
	   CALL INIT_EMPTY_CONTEXT(cx_buff)
*       ignore status ... always OK
	   CALL CONSTANT_CONTEXT("0", cx_buff, cx_buff, status)
	   cx_color = cx_buff
	   mr_color = dummy_mr
        ELSE
	   GOTO 5200 
        ENDIF

* initialize
	nline_in_mem = 0
	pxlim = qual_given( slash_xlimits )
	pylim = qual_given( slash_ylimits )
	ptitl = qual_given( slash_title )
	user_title = ptitl .GT. 0
	cx_x = cx_list(1)
	cx_y = cx_list(2)
	mr_x = mr_list(1)
	mr_y = mr_list(2)
	plot_mem_used = 0
	lox = arbitrary_large_val4
	hix = arbitrary_small_val4
	loy = arbitrary_large_val4
	hiy = arbitrary_small_val4
!	ind_dim = plot_axis(1)
	stack_up = .FALSE.

* perform sanity checks on the arguments
* ... check the X coordinate variable for shape
	CALL GET_CX_DIMS( cx_x, ndim, dims)
	IF (ndim .EQ. 0) GOTO 5210		! npoints = 1 !!! yuk!
	IF (ndim .GT. 2) GOTO 5210		! could someday be relaxed ...
* ... check the Y coordinate variable for conformability
	CALL GET_CX_DIMS( cx_y, nfdim, fdims)
	IF (ndim .NE. nfdim) GOTO 5210
	DO 100, idim = 1, ndim
	   IF (dims(idim) .NE. fdims(idim)) GOTO 5210
	   IF ( CX_DIM_LEN(dims(idim),cx_x)
     .    .NE.  CX_DIM_LEN(dims(idim),cx_y) ) GOTO 5210
 100	CONTINUE
* ... check the color value array for consistent structure
	oneD_form = ndim .EQ. 1
	CALL GET_CX_DIMS( cx_color, nfdim, fdims)
	IF (nfdim .GT. 1) GOTO 5210

* determine coordinate axes for polygon coords and fill color values
	IF (oneD_form) THEN
* 1-dimensional form of command
	   coord_dim = dims(1)	! axis of points in the polygon(s)
	   color_dim = coord_dim
	ELSE
* 2-dimensional form of command
* ... does the color value axis match one of the coordinate array axes?
	   IF (nfdim .EQ. 0) THEN
	      ambiguous_coord_axis = .TRUE.
	   ELSE
	      DO 105 idim = 1, ndim
	         IF (dims(idim) .EQ. fdims(1)) THEN
	            ambiguous_coord_axis = .FALSE.
	            GOTO 106
	         ENDIF
 105          CONTINUE
	      ambiguous_coord_axis = .TRUE.
	   ENDIF
 106	   IF (ambiguous_coord_axis) THEN
* ... abmiguous 2D form since since color_array is a scalar
	      IF (coord_ax .GT. 0 ) THEN
* ... the user has specified the coordinate axis (e.g. "/COORD_AX=J")
	         CALL EQUAL_STRING( cmnd_buff(qual_start(coord_ax):
     .					      qual_end  (coord_ax)),
     .				    buff, status )
                 IF ( status .NE. ferr_ok ) RETURN
                 DO 120 coord_dim = 1,4
	            IF (ww_dim_name(coord_dim) .EQ. buff(1:1)
     .		   .OR. ss_dim_name(coord_dim) .EQ. buff(1:1) ) THEN
*      ... is the specified axis an axis of the arrays?
	               DO 110 idim = 1, 2
 110		       IF (coord_dim .EQ. dims(idim)) GOTO 150
	               GOTO 5300	! invalid coordinate dim given
	            ENDIF
 120	         CONTINUE
	         GOTO 5300	! coord dim wasn't "X","Y","Z", or "T"
	      ELSE
* ... the user has NOT specified the coordinate axis
	         coord_dim = dims(2)
	         CALL WARN("Ambiguous axes in 2D coordinate arrays")
	         CALL WARN("Continuing as /COORD_AX="//
     .				ww_dim_name(coord_dim))
	      ENDIF
 150	      IF (dims(1) .EQ. coord_dim) THEN
	         color_dim = dims(2)
	      ELSE
	         color_dim = dims(1)
	      ENDIF
	   ELSE
* color fill value array is 1D and matches an axis of the coordinate arrays
	      color_dim = fdims(1)
	      IF (dims(1) .EQ. color_dim) THEN
	         coord_dim = dims(2)
	      ELSE
	         coord_dim = dims(1)
	      ENDIF
	   ENDIF
	ENDIF

* set flag indicating a 1D plot is on the screen
	IF (.NOT.overlay) onedee_on = .TRUE.

* prepare for possible transpose of X and Y axes
	IF ( transpz ) THEN
	   y_ax  = 'X'
	   x_ax  = 'Y'
	   y_lab = ppl_xlab
	   x_lab = ppl_ylab
	   y_len   = xlen
	   x_len   = ylen
	ELSE
	   y_ax  = 'Y'
	   x_ax  = 'X'
	   y_lab = ppl_ylab
	   x_lab = ppl_xlab
	   x_len   = xlen
	   y_len   = ylen
	ENDIF

* * * * * * *    - - - LOAD DATA - - -    * * * * * * * * *
* allocate modifiable context buffers for data extraction
	CALL STACK_PTR_UP( cx_stack_ptr, max_context, status )
	IF ( status .NE. ferr_ok ) RETURN
	cx_buff_x = cx_stack_ptr
	CALL STACK_PTR_UP( cx_stack_ptr, max_context, status )
	IF ( status .NE. ferr_ok ) RETURN
	cx_buff_y = cx_stack_ptr
	CALL TRANSFER_CONTEXT( cx_x, cx_buff_x )
	CALL TRANSFER_CONTEXT( cx_y, cx_buff_y )
	stack_up = .TRUE.

* set start indices for extraction
	IF (oneD_form) THEN
	   end  = 0
	ELSE
	   xrow = cx_lo_ss(cx_x,color_dim) - 1
	   yrow = cx_lo_ss(cx_y,color_dim) - 1
	ENDIF

* tell PLOT+ the bad data flags used
	WRITE ( ppl_buff, 3005 ) mr_bad_data( mr_x ), x_ax
	CALL PPLCMD ( from, line, 0, ppl_buff, 1, 1 )
	WRITE ( ppl_buff, 3005 ) mr_bad_data( mr_y ), y_ax
	CALL PPLCMD ( from, line, 0, ppl_buff, 1, 1 )
 	WRITE ( ppl_buff, 3005 ) mr_bad_data( mr_color ), 'Z'
	CALL PPLCMD ( from, line, 0, ppl_buff, 1, 1 )
 3005	FORMAT ( 'LIMITS ',G15.8,1X,A1,'EQ' )
	
* loop through all the polygons variables loading them into PPLUS
* construct a funky FORTRAN 77 DO-WHILE loop
	npolygon = 0
* ---- 
200	CONTINUE

* ... prepare the context buffer to point to the next polygon 
	   IF ( oneD_form ) THEN
*    ... coords are 1D arrays of p1,p1,p1,p1,BAD,p2,p2,p2,p2,BAD,...
	      cx_lo_ss(cx_buff_x,coord_dim) =
     .	      cx_lo_ss(cx_buff_x,coord_dim) + end   ! advance past end of last
	      cx_lo_ss(cx_buff_y,coord_dim) =
     .	      cx_lo_ss(cx_buff_y,coord_dim) + end
	      IF (cx_lo_ss(cx_buff_x,coord_dim)
     .	     .GE. cx_hi_ss(cx_buff_x,coord_dim)) GOTO 290 ! exit DO_WHILE loop
	   ELSE
*    ... coords are 2D arrays of color_dim X coord_dim
	      xrow = xrow + 1  ! advance to next polygon in the 2D array
	      yrow = yrow + 1  ! advance to next polygon in the 2D array
	      IF (xrow .GT. cx_hi_ss(cx_x,color_dim)) GOTO 290	! exit the DO_WHILE loop
	      cx_lo_ss(cx_buff_x,color_dim) = xrow
	      cx_hi_ss(cx_buff_x,color_dim) = xrow
	      cx_lo_ss(cx_buff_y,color_dim) = yrow
	      cx_hi_ss(cx_buff_y,color_dim) = yrow
	   ENDIF

* ... extract 1-dimensional arrays -- possibly with embedded missing values
* ... x coordinates
	   CALL EXTRACT_LINE (	cx_buff_x,
     .				memory( 1, mr_blk1(mr_x) ),
     .				mr_x,
     .				x_dat,
     .				coord_dim,
     .				ndim,
     .				npts,
     .				status	)
	   IF ( status .NE. ferr_ok ) GOTO 5000
* ... y coordinates
	   CALL EXTRACT_LINE (	cx_buff_y,
     .				memory( 1, mr_blk1(mr_y) ),
     .				mr_y,
     .				y_dat,
     .				coord_dim,
     .				ndim,
     .				npts,
     .				status	)
	   IF ( status .NE. ferr_ok ) GOTO 5000

* ... locate the (next) contiguous segment in the arrays
*    ...search for start of segment
	   missing_polygon = .false.
	   start = npts
	   end   = npts
	   DO 210 i = 1, npts
	      IF ( x_dat(i) .NE. mr_bad_data(mr_x) 
     .       .AND. y_dat(i) .NE. mr_bad_data(mr_y) ) THEN
	         start = i
	         GOTO 220
	      ENDIF
 210	   CONTINUE
	   GOTO 240   ! This polygon is all missing in X - omit		
*    ... find the end of the segment
 220	   DO 230 i = start, npts
	      IF ( x_dat(i) .EQ. mr_bad_data(mr_x) 
     .       .OR.  y_dat(i) .EQ. mr_bad_data(mr_y) ) THEN
	         end = i - 1
	         GOTO 250
	      ENDIF
 230	   CONTINUE

 240	   IF (start .GE. end ) missing_polygon = .true.

* ... check for adequate PLOT+ memory
 250	   npts = end-start+1
	   plot_mem_used = plot_mem_used + npts
	   IF ( plot_mem_used .GT. 0.99*nsize ) GOTO 5150

* ... maintain check on x data range for PLOT+ auto-scaling
	   CALL MINMAX( x_dat(start), npts, mr_bad_data(mr_x), lo, hi)
	   IF (lo .LT. lox) lox = lo
	   IF (hi .GT. hix) hix = hi
	   CALL MINMAX( y_dat(start), npts, mr_bad_data(mr_y), lo, hi)
	   IF (lo .LT. loy) loy = lo
	   IF (hi .GT. hiy) hiy = hi

* ... pass the data to PLOT+

	   if (overlay) SOVER = .true.

* ... let PPL know what's going on
	   makep = 'FILLPOL'

* ... setup for possible overlay on a time axis plot
	   IF (overlay .AND. has_time_axis) THEN
	      dt_min  = saved_dt_min
	      t1_date = saved_t1_date
	   ELSE
              dt_min  = 1.0
              t1_date = ' '
	   ENDIF

	   IF (missing_polygon) THEN
	      CALL PPLLDX(0,mr_bad_data(mr_x),mr_bad_data(mr_y),1,
     .    	          t1_date,' ',dt_min)	
	   ELSE IF ( transpz ) THEN
	      CALL PPLLDX(0,y_dat(start),x_dat(start),npts,
     .		          t1_date,' ',dt_min)
	   ELSE
	      CALL PPLLDX(0,x_dat(start),y_dat(start),npts,
     .			  t1_date,' ',dt_min)
	   ENDIF

* ... increment number of lines on plot
!	   nline_on     = nline_on     + 1
	   nline_in_mem = nline_in_mem + 1

* ... assign the same line style ("1") to all polygons
CCC	   CALL LINE_STYLE( symbol, line_type, nline_in_mem, 1 )

	npolygon = npolygon + 1
	GOTO 200	! end of DO-WHILE loop
 290	IF (npolygon .EQ. 0 ) GOTO 5230

* NOW set the line style
	CALL LINE_STYLE( symbol, line_type, 1, 1 )

* release context stack buffers
	cx_stack_ptr = cx_stack_ptr - 2
	stack_up = .FALSE.

* pass the color value array to PLOT+ (use x_dat as a buffer)
	if (nmv .eq. 2) then
	   ncolor = 1
	   x_dat(1) = 0.0
	else
	   CALL EXTRACT_LINE (	cx_color,
     .				memory( 1, mr_blk1(mr_color) ),
     .				mr_color,
     .				x_dat,
     .				color_dim,
     .				ndim,
     .				ncolor,
     .				status	)
	   IF ( status .NE. ferr_ok ) GOTO 5000
	endif
	IF (ncolor .EQ. 1) THEN
	   DO 300 ipoly = 1, npolygon
 300	   x_dat(ipoly) = x_dat(1)	! single color applied to all pol'gs
	   ncolor = npolygon
	ENDIF
	IF (ncolor .NE. npolygon) GOTO 5220	! color array length wrong

* pass back a summary about the range of colors
	no_range = NO_LINE_RANGE(x_dat,
     .				 ncolor, mr_bad_data(mr_color), only_val )

	if (overlay) SOVER = .true.
	makep = 'FILLPOL'

	CALL PPLLDX(2,x_dat,x_dat,ncolor,t1_date,' ',dt_min)

* turn on lon/lat axis formatting for polygon plots, otherwise off

	IF (.NOT.overlay) THEN
	   iunits = TM_UNIT_ID(VAR_UNITS(cx_x))
	   IF (iunits .EQ. 4) THEN
	      CALL PPLCMD ( from, line, 0, 'XFOR (I7,''LONE'')', 1, 1 )
	   else
	      CALL PPLCMD ( from, line, 0, 'XFOR 0', 1, 1 )
	   endif
	   
	   iunits = TM_UNIT_ID(VAR_UNITS(cx_y))
	   IF (iunits .EQ. 4) THEN
	      CALL PPLCMD ( from, line, 0, 'YFOR (I7,''LAT'')', 1, 1 )
	   else
	      CALL PPLCMD ( from, line, 0, 'YFOR 0', 1, 1 )
	   endif
	ENDIF
	

* axis scaling -- only applies if it is a fresh plot (not an overlay)
* most of the work is avoiding an error if there is no data range with
* which to make a scale ...
	IF ( .NOT.overlay ) THEN
	   delta = unspecified_val8
* ... X axis
	   IF ( pxlim .GT. 0 ) THEN   ! explicit /XLIMITS 10/93
	      CALL EQUAL_RANGE( cmnd_buff(qual_start(pxlim):qual_end(pxlim)),
     .				x_dim, vmin, vmax, delta,
     .				formatted, status )
	      IF ( status .NE. ferr_OK ) GOTO 5000
              CALL AXIS_ENDS(x_ax,coord_dim,cx_grid(cx_x),
     .			     vmin,vmax,delta)
	   ELSEIF (lox .GE. hix) THEN
	      IF (lox .EQ. arbitrary_large_val4) lox = 1.0
* ... no data range -- provide artifical axis limits
              CALL AXIS_ENDS(x_ax,coord_dim,cx_grid(cx_x),
     .			DBLE(lox-1.0),DBLE(lox+1.0), 0.5D0)
	   ELSE
              CALL AXIS_ENDS(x_ax,coord_dim,cx_grid(cx_x),
     .			DBLE(lox),DBLE(hix), delta)

	   ENDIF
* ... Y axis
	   IF ( pylim .GT. 0 ) THEN   ! explicit /XLIMITS 10/93
	      CALL EQUAL_RANGE( cmnd_buff(qual_start(pylim):qual_end(pylim)),
     .				y_dim, vmin, vmax, delta,
     .				formatted, status )
	      IF ( status .NE. ferr_OK ) GOTO 5000
              CALL AXIS_ENDS(y_ax,coord_dim,cx_grid(cx_y),
     .			     vmin,vmax,delta)
	   ELSEIF (loy .GE. hiy) THEN
	      IF (loy .EQ. arbitrary_large_val4) loy = 1.0
* ... no data range -- provide artifical axis limits
              CALL AXIS_ENDS(y_ax,coord_dim,cx_grid(cx_y),
     .			DBLE(loy-1.0),DBLE(loy+1.0), 0.5D0)
	   ELSE
              CALL AXIS_ENDS(y_ax,coord_dim,cx_grid(cx_y),
     .			DBLE(loy),DBLE(hiy), delta)

	   ENDIF
	ENDIF

	IF ( no_labels ) RETURN

* * * * * * *    - - - LABELS - - -   * * * * * * * * *
* if we are starting from a blank screen each axis is labelled with the units
* of its coordinate variable
* The plot title is derived from the color value variable (unless over-ridden
* by the /TITLE qualifier) and it is placed in a key for overlays or as a
* title for a fresh plot.
	IF ( overlay ) THEN
	   CALL OV_GRP_CX_DOC( cx_color, 1, cx_xtra_doc )
	   IF ( user_title ) THEN
	      CALL EQUAL_STR_LC(
     .			cmnd_buff(qual_start(ptitl):qual_end(ptitl)),
     .                  plot_title, status )
              IF ( status .NE. ferr_ok ) GOTO 5000
	      slen = TM_LENSTR1( plot_title )
	   ELSE
	      plot_title = KEY_STRING( cx_color, cx_xtra_doc, slen )
	   ENDIF
	   CALL KEY_LOC( overlay, xkeypos, ykeypos, xkeysize, ykeysize,
     .			 lab_ht, style, justify )
*          Key uses line 1
	   CALL PUT_KEY( 1, plot_title(:slen),
     .			 xkeypos, ykeypos, xkeysize, ykeysize,
     .			 lab_ht, style, lab_loc_absolute, lab_loc_absolute )
	ELSE
* ... basic position and data set labels
	   CALL DISP_LABELS( cx_plot, unspecified_name4 )
* ... title
	   IF ( user_title ) THEN     ! user over-ride of title
	      CALL EQUAL_STR_LC
     .		( cmnd_buff(qual_start(ptitl):qual_end(ptitl)),
     .            plot_title, status )
              IF ( status .NE. ferr_ok ) GOTO 5000
	      slen = TM_LENSTR1( plot_title )
	   ELSE
	      plot_title = FULL_VAR_TITLE( cx_color, slen )
	   ENDIF
	   CALL BOX_LABEL( ppl_title,
     .			 plot_title(:slen),
     .			 0.0, 0.0, 0.75*xlen, letsize_title,
     .			 letsize_title, ppl_centered,
     .			 lab_loc_absolute, lab_loc_absolute )
* ... label axis labels:  units from the polygon coordinates

	   plot_title = FULL_VAR_TITLE( cx_x, slen )

C	   plot_title = VAR_UNITS( cx_x )
C	   slen = TM_LENSTR1( plot_title )
	   CALL BOX_LABEL( x_lab,
     .			 plot_title(:slen),
     .			 0.0, 0.0, 0.6*x_len, letsize_label,
     .			 letsize_label, ppl_centered,
     .			 lab_loc_absolute, lab_loc_absolute )

	   plot_title = FULL_VAR_TITLE( cx_y, slen )

C	   plot_title = VAR_UNITS( cx_y )
C	   slen = TM_LENSTR1( plot_title )
	   CALL BOX_LABEL( y_lab,
     .			 plot_title(:slen),
     .			 0.0, 0.0, 0.6*y_len, letsize_label,
     .			 letsize_label, ppl_centered,
     .			 lab_loc_absolute, lab_loc_absolute )
	 ENDIF

* successful completion
 1000	status = ferr_ok
	RETURN

* error exit
 5000	CALL PPLCMD ( from, line, 0, 'NLINES', 1, 1 )	! wipe buffers clean
	IF (stack_up) cx_stack_ptr = cx_stack_ptr - 2
	RETURN
 5150	buff = LEFINT( INT(0.99*nsize), slen )
	CALL ERRMSG( ferr_prog_limit, status,
     .		'Plot buffer size is: '//buff(:slen), *5000 )
 5200	CALL ERRMSG( ferr_invalid_command, status,
     .		'POLYGON may use 2 args (xpts,ypts) with /LINE only,' //
     .          ' or 3 args (xpts,ypts,color_value)',
     .		*5000)
 5210	CALL ERRMSG( ferr_invalid_command, status,
     .		'illegal dimensions for polygon coordinate(s) or values',
     .		*5000)
 5220	buff2 = LEFINT(npolygon, slen2)
	buff  = LEFINT(ncolor, slen) 
	CALL ERRMSG( ferr_invalid_command, status,
     .		'no. of values ('//buff(:slen)//') '//
     .		'doesnt match no. of polygons ('//buff2(:slen2)//')', 
     .		*5000)
 5230	CALL ERRMSG( ferr_invalid_command, status,
     .		'zero polygons specified',
     .		*5000)
 5300	CALL ERRMSG( ferr_invalid_command, status,
     .		'invalid argument to /COORD_AX=',
     .		*5000)

	END
