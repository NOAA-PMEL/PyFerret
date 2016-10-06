	SUBROUTINE POLYGON_SET_UP
     .			      ( memory,
     .				overlay, transpz, coord_ax, no_labels,
     .				no_range, only_val,
     .				symbol, line_type, use_line,
     .				is_logx, is_logy, 
     .				mr_list, cx_list, nmv,
     .				is_modx,
     .				do_mod_minus, 
     .				do_mod_plus,
     .				add_lon,
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
* V510: *sh* 3/00 - trap if number of polygons is too large 
*		  - new arg use_line for change in arguments to LINE_STYLE
*	5/00 - added /HLIMITS and /VLIMITS (denigrating /xlimits/ylimits)
* V510: *sh* 7/00 - added arg to EQUAL_RANGE 
* V530 *sh* 1/01 - change "denigrated" to "deprecated"
*	    1/01 - remove makep setting so segments are not pre-activated
* V530 *ACM* 1/03  cal_id for alternative calendar axes.
* V530 *ACM* 5/09  add arguments (unused here) to line_style call. 
* V533 *acm* 6/01 - for LOG plots add logicals is_logx, is_logy.  set to plot
*                   log of depth as inverse axis
* V533 *acm* 9/01 - changes for dynamic pplus buffer.  Rename NSIZE to pplmem_nsize.
*                   call PPLLDX via pplldx_envelope.c which checks pplmem_nsize 
*                   and allocates more memory if needed.  
*      *acm* 10/01  remove VMS includes
* v540 *acm* 12/01  IF (missing_polygon) THEN don't call PPLLDX_envelope; calling
*                   with missing data for time-axis data results in trying to 
*                   compute a time axis (elsewhere) with out-of-range values.
* v541 *acm*  3/02  The above fix is too broad.  Apply it only to time-axis 
*                   overlays.
* v541 *acm*  3/02  Compute and allocate the ammount of PPLUS memory that'll
*                   be needed for this plot, before the loop that loads polygons.
* v541 *acm*  3/02  set twodee_on to .TRUE. (as in disp_init) so we can overlay on 
*                   polygon plots
* v541 *acm*  4/02  If we have a depth axis, multiply the corresponding 
*                   vertices by -1.  Else the polygons arent plotted.
* v541 *acm*  8/02  For "No Valid Data" the test is: 
*                   IF (no_range .AND. (only_val.EQ.mr_bad_data(mr_color))
* V550 *acm* 11/02  When its a 1-d form the ppl memory allocation was wrong.
*                   (Input is 1D arrays of p1,p1,p1,p1,BAD,p2,p2,p2,p2,BAD,..)
* V552 *acm*  3/03  Send the key string to key_loc since it may be multi-line
*                   Also, set strings longer
* V553 *acm*  6/03  For polygons, store the # of points in each polygon in the PPLUS
*                   memory buffer, rather than in lleng.  This way the number of
*                   polygons that may be plotted is unlimited. Mark this with the
*                   value of icode (first argument to pplldx_envelope)
* v554 *acm* 12/03  Bug fix: for 1-D form of polygon command, save values of context 
*                   so they can be properly restored. (ss_x_save, ss_y_save)
* V580 *acm* 11/04  Check calendar id only for 2-D polygon form; 1-D polygon data are
*                   just lists on abstract axis.  Fix for bug 1066
* V580 *acm* 12/04  Correct the syntax sending XFOR LONE to pplus when we 
*                   turn on lon/lat formatting.
* V600 *acm*  11/05 Fix for bug 1130, 1349: axis labels need lat/lon labels,
*                   fixes to the precision
* V62  *acm* 5/09   Fixing bug 1661: if the plot is TRANSPOSED, then multiply 
*                   vertices of the opposite axis by -1 
* V630  *acm* 9/09 Introduction of syntax for variance-based and histogram levels
* V650 *acm*  3/10  Check calendar id only if polygon data is not a list on an
*                   abstract time axis.  Fix for bug 1722.
* V671 *acm*  5/11  For the 1D form of polygon plots, the check for adequate 
*                   PLOT+ memory was not adequate - we need x,y, and variable 
*       *acm* 4/12 6D Ferret: time axis may be in t or f direction.
* V6.85 *acm* 9/13 Change to CALL LINE_STYLE, implementing new PLOT/COLOR w/o 
*                  argument for all-color line plot
* V686 *acm* 11/13 Allow symbol names to be up to 120 characters long
* V691  *acm*  5/14 Clean up temp memory used for computing histogram levels
* V695  *acm*  9/15 send REAL*4 rbad to compare with lev_max, lev_min inside compute_mnstd
* V698  *acm*  3/16 #2396: On a POLYGON/OVERLAY with no /TITLE given, if there is 
*                   no 3rd argument (colors), make nominal title
* V698  4/16 *acm* New PLOT/SKIP= means change in call sequence for LINE_STYLE
* V701  *acm* 8/16 Fix ticket 2463: single-valued data with a /LEVELS setting
* V702  *acm* 10/16 Test the results of MINMAX routine before use; if all data is 
*                   all-missing it returns arbitrary_large_val4.

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
        include 'PPLDAT.INC'
	include 'parampl5_dat.decl'
	include 'PARAMPL5.DAT'	! with NPOLYGONS
        include 'switch_inc.decl'
        include 'SWITCH.INC'
        include 'pltl_inc.decl'
        include 'PLTL.INC'
	include 'vector_inc.decl' ! with TRANSPZ
        include 'VECTOR.INC'


* calling argument declarations:
	LOGICAL		overlay, transpz, no_labels, no_range, use_line,
     .			is_logx, is_logy, is_modx, do_mod_minus, do_mod_plus
	INTEGER		symbol, line_type, coord_ax, iunits, do_dash,
     .			nmv, mr_list( nmv ), cx_list( nmv ), status
	REAL		memory( mem_blk_size, max_mem_blks )
	REAL		x_dat(*), y_dat(*)     ! dynamic space allocation 
	REAL            only_val, dashstyle(4), add_lon

* internal variable declarations:
	LOGICAL         missing_polygon, first_polygon, use_nice
	LOGICAL		NO_LINE_RANGE, ambiguous_coord_axis,
     .			formatted, user_title, oneD_form, stack_up,
     .			x_is_log, y_is_log
	CHARACTER*2048	KEY_STRING, FULL_VAR_TITLE, plot_title
* 3/99 *kob* - upped VAR_UNITS from 32 to 64 chars
	CHARACTER	VAR_UNITS*64, LEFINT*6, TM_GET_CALENDAR_NAME*32, 
     .			x_ax*1, y_ax*1,buff*8, buff2*4, t1_date*14,
     .                  cal_name*32, cal_name_saved*32, buff3*16,
     .			buff4*16, LEFT_REAL*16, fmt*4, buffer*40, buffsym*30
	INTEGER         TM_UNIT_ID
	INTEGER		TM_LENSTR1, CX_DIM_LEN,	TM_GET_CALENDAR_ID, 
     .			STR_SAME,
     .			mr_x, cx_x, mr_y, cx_y, cx_color, mr_color,
     .			slen, slen2, x_lab, y_lab, nline_in_mem, i,
     .			npoly, ncolor, ipoly, xrow, yrow, start, end,
     .			npts, idim, color_dim, coord_dim,
     .			style, justify, ndim, dims(nferdims), nfdim,
     .			fdims(nferdims), plot_mem_used, pxlim, pylim, ptitl,
     .			cx_buff_x, cx_buff_y,
     .			phlim, pvlim, pxdenig, pydenig,
     .                  cal_id, tax, x_axtyp, y_axtyp, n,
     .                  ss_x_save, ss_y_save, hblk1, mvh_temp, ier, nc, skipsym

	REAL		x_len, y_len, lo, hi, lox, hix, loy, hiy,
     .			xkeypos, ykeypos, xkeysize, ykeysize, lab_ht,
     .			dt_min, epsilon, small, scale,
     .			xptsmin, xptsmax, xptsmean, xdif, xmeanlog,
     .			yptsmin, yptsmax, yptsmean, ydif, ymeanlog,
     .                  first, last

	REAL*8		vmin, vmax, delta, ind_min, ind_max, dep_min,
     .			dep_max
	REAL*4 rbad

* local parameter declarations:
	LOGICAL		range_rqd
	PARAMETER     ( range_rqd = .TRUE. )
	PARAMETER     ( epsilon = 1.E-45 )


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
	pxlim = qual_given( slash_xlimits ) ! deprecated
	pylim = qual_given( slash_ylimits ) ! deprecated
	phlim = qual_given( slash_hlimits )
	pvlim = qual_given( slash_vlimits )
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
c	ind_dim = plot_axis(1)
	stack_up = .FALSE.
	cal_id = 0
	x_axtyp = 1
	y_axtyp = 1

* *acm* 3/02  set flag indicating a 2D plot is on the screen
*             (so we can overlay other plots on a polygon plot)
	twodee_on = .TRUE.

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

        IF (.NOT. overlay) has_time_axis = .FALSE.  ! ??

* prepare for possible transpose of X and Y axes
	IF ( transpz ) THEN
	   y_ax  = 'X'
	   x_ax  = 'Y'
	   y_lab = ppl_xlab
	   x_lab = ppl_ylab
	   y_len   = xlen
	   x_len   = ylen
	   x_is_log = is_logy
           y_is_log = is_logx

	   IF (pxlim .GT. 0) THEN
	      pxdenig = pxlim
	   ELSE
	      pxdenig = pvlim
	   ENDIF
	   IF (pylim .GT. 0) THEN
	      pydenig = pylim
	   ELSE
	      pydenig = phlim
	   ENDIF
	ELSE
	   y_ax  = 'Y'
	   x_ax  = 'X'
	   y_lab = ppl_ylab
	   x_lab = ppl_xlab
	   x_len   = xlen
	   y_len   = ylen
	   x_is_log = is_logx
           y_is_log = is_logy

	   IF (pxlim .GT. 0) THEN
	      pxdenig = pxlim
	   ELSE
	      pxdenig = phlim
	   ENDIF
	   IF (pylim .GT. 0) THEN
	      pydenig = pylim
	   ELSE
	      pydenig = pvlim
	   ENDIF
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


* need number of polygons and points in each. On the first polyon we will
* check that we have enough PPLUS memory and allocate more if needed.

        first_polygon = .TRUE.

* ...  point to the first polygon 
       IF ( oneD_form ) THEN
*    ... coords are 1D arrays of p1,p1,p1,p1,BAD,p2,p2,p2,p2,BAD,...
*    ... we will find npoly as we go through the list
          npoly = 0
	ELSE
*    ... coords are 2D arrays of color_dim X coord_dim
           npoly = cx_hi_ss(cx_x,color_dim) - cx_lo_ss(cx_x,color_dim) + 1
	ENDIF

* loop through all the polygons variables loading them into PPLUS
* construct a funky FORTRAN 77 DO-WHILE loop
* ---- 
        IF (oneD_form) THEN
           ss_x_save = cx_lo_ss(cx_buff_x,coord_dim)
           ss_y_save = cx_lo_ss(cx_buff_y,coord_dim)
        ENDIF

200	CONTINUE

* ... prepare the context buffer to point to the next polygon 
	   IF ( oneD_form ) THEN
*    ... coords are 1D arrays of p1,p1,p1,p1,BAD,p2,p2,p2,p2,BAD,...
	      cx_lo_ss(cx_buff_x,coord_dim) =
     .	       cx_lo_ss(cx_buff_x,coord_dim) + end   ! advance past end of last
	      cx_lo_ss(cx_buff_y,coord_dim) =
     .	       cx_lo_ss(cx_buff_y,coord_dim) + end
	      IF (cx_lo_ss(cx_buff_x,coord_dim)
     .	     .GE. cx_hi_ss(cx_buff_x,coord_dim)) THEN
                
 	         cx_lo_ss(cx_buff_x,coord_dim) = ss_x_save   ! restore what it was
	         cx_lo_ss(cx_buff_y,coord_dim) = ss_y_save   ! so we can count npts
                                                             ! in the 700 loop below
                 GOTO 290 ! exit DO_WHILE loop
              ENDIF

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

*  For 2d form, npts in each polygon; check PPLUS memory and if needed, 
*  allocate more.  Multiply by 2 because we have X and Y vertices of 
*  polygons, and by 2 again because the call below to 
*       PPLLDX_envelope(5,x_dat,x_dat,ncolor...  
*  loads polygon data after all the previous data.

	   IF (first_polygon .AND. .NOT.oneD_form) THEN
              plot_mem_used = 4* npts* npoly
              CALL reallo_envelope(plot_mem_used)

              first_polygon = .FALSE.
              plot_mem_used = 0
              npoly = 0
           ENDIF

* If one of the axes is a depth axis, multiply the corresponding 
* vertices by -1.  Else they don't show up on the plot.
* xlo, xhi, ylo, yhi are from AXIS.INC

           IF (transpz) THEN
              IF ( xlo .GT. xhi )  THEN
                 DO 201 i = 1, npts
                    IF ( y_dat(i) .NE. mr_bad_data(mr_y) )
     .                y_dat(i) = -1.* y_dat(i)
  201            CONTINUE
              ENDIF
   
              IF ( ylo .GT. yhi )  THEN
                 DO 202 i = 1, npts
                    IF ( x_dat(i) .NE. mr_bad_data(mr_x) )
     .                x_dat(i) = -1.* x_dat(i)
  202            CONTINUE
              ENDIF

   	   ELSE

      	      IF ( xlo .GT. xhi )  THEN
                 DO 203 i = 1, npts
                    IF ( x_dat(i) .NE. mr_bad_data(mr_x) )
     .                x_dat(i) = -1.* x_dat(i)
  203            CONTINUE
              ENDIF
   
              IF ( ylo .GT. yhi )  THEN
                 DO 204 i = 1, npts
                    IF ( y_dat(i) .NE. mr_bad_data(mr_y) )
     .                y_dat(i) = -1.* y_dat(i)
  204            CONTINUE
              ENDIF
	   ENDIF  

	   IF (add_lon .NE. 0.) THEN
	      DO 205 i = 1, npts
	         IF ( x_dat(i) .NE. mr_bad_data(mr_x) ) 
     .              x_dat(i) = add_lon + x_dat(i)
  205	      CONTINUE
           ENDIF

* When using log axis transform to log(x).  Scale the points of the
* polygon within the log axis range xlo, xhi to keep something like
* the original shape of the polygon.

	   IF (x_is_log) THEN

*  Find min and max x points in the polygon shape (linear units)

	      xptsmin = ABS(mr_bad_data(mr_x))
	      xptsmax = -1.* xptsmin

	      DO 400 i = 1, npts
		 IF (x_dat(i) .NE. mr_bad_data(mr_x)) THEN 
		    xptsmin = MIN(xptsmin, x_dat(i))
		    xptsmax = MAX(xptsmax, x_dat(i))
		 ENDIF
  400         CONTINUE

	      xptsmean = (xptsmin+xptsmax)/2.

*  Convert the x midpoint to log units, and put the other x points
*  on the page scaled by the axis ends xlo and xhi

	      IF (xptsmean .GT. epsilon) THEN
		 xmeanlog = LOG10(xptsmean)
		 scale = 0.1* (xhi-xlo)	! log axis units

		 DO 410 i = 1, npts
		    xdif = scale*(x_dat(i) - xptsmean)
		    x_dat(i) = xmeanlog + xdif
  410		 CONTINUE
	      ELSE
		 small = xptsmean
		 GO TO 5600
	      ENDIF
	   ENDIF

* When using log axis transform to log(y).  Scale the points of the
* polygon within the log axis range ylo, yhi to keep something like
* the original shape of the polygon.

	   IF (y_is_log) THEN

*  Find min and max y points in the polygon shape (linear units)

	      yptsmin = ABS(mr_bad_data(mr_y))
	      yptsmax = -1.* yptsmin

	      DO 500 i = 1, npts
		 IF (y_dat(i) .NE. mr_bad_data(mr_y)) THEN 
		    yptsmin = MIN(yptsmin, y_dat(i))
		    yptsmax = MAX(yptsmax, y_dat(i))
		 ENDIF
  500         CONTINUE

	      yptsmean = (yptsmin+yptsmax)/2.

*  Convert the y midpoint to log units, and put the other Y points
*  on the page scaled by the axis ends ylo and yhi

	      IF (yptsmean .GT. epsilon) THEN
		 ymeanlog = LOG10(yptsmean)
		 scale = 0.1* (yhi-ylo)	! log axis units

		 DO 510 i = 1, npts
		    ydif = scale*(y_dat(i) - yptsmean)
		    y_dat(i) = ymeanlog + ydif
  510		 CONTINUE
	      ELSE
		 small = yptsmean
		 GO TO 5600
	      ENDIF
	   ENDIF

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

           IF (oneD_form) THEN
              plot_mem_used = plot_mem_used + 3*npts
              CALL reallo_envelope(plot_mem_used)
           ENDIF

c	   IF ( npoly         .GE. NPOLYGON   ) GOTO 5160

* ... maintain check on x data range for PLOT+ auto-scaling
	   CALL MINMAX( x_dat(start), npts, mr_bad_data(mr_x), lo, hi)
	   IF (lo .NE. arbitrary_large_val4) THEN ! if it is arb-large, then all missing
	      IF (lo .LT. lox) lox = lo
	      IF (hi .GT. hix) hix = hi
	   ENDIF
	   CALL MINMAX( y_dat(start), npts, mr_bad_data(mr_y), lo, hi)
	   IF (lo .NE. arbitrary_large_val4) THEN ! if it is arb-large, then all missing
	      IF (lo .LT. loy) loy = lo
	      IF (hi .GT. hiy) hiy = hi
	   ENDIF

* ... pass the data to PLOT+

	   if (overlay) SOVER = .true.

* ... let PPL data loader know what's going on (kludge)
	   makep = 'FILLPOL'

* ... setup for possible overlay on a time axis plot  ? calendar_id here?
	   IF (overlay .AND. has_time_axis  ) THEN
	      dt_min  = saved_dt_min
	      t1_date = saved_t1_date
              IF ( .NOT. oneD_form) THEN
              
*  Check for calendar mismatch for time- overlay plots
*  (but only for 2-D polygon form; 1-D polygon data are
*  just lists on abstract axis.)
*  If the axis is 7 (abstract) then dont do the check either.
                 tax = grid_line(f_dim,cx_grid(cx_x ))
                 IF (tax.EQ.mnormal) tax = grid_line(t_dim,cx_grid(cx_x ))
		 IF (STR_SAME(line_name(tax),'ABSTRACT') .NE. 0) THEN 
                    cal_name = line_cal_name(tax)
                    cal_id = TM_GET_CALENDAR_ID ( cal_name )
                    IF (cal_id .NE. saved_calendar_id) GO TO 5360
                 ENDIF
              ENDIF
	   ELSE
              dt_min  = 1.0
              t1_date = ' '
	   ENDIF

	   IF (missing_polygon) THEN
! for time series overlays, dont load the data
              IF (.NOT. (has_time_axis .AND. overlay)) 
     .    	  CALL PPLLDX_envelope(3,mr_bad_data(mr_x),
     .            mr_bad_data(mr_y),1,t1_date,' ',dt_min,plot_mem_used)
	   ELSE IF ( transpz ) THEN
	      CALL PPLLDX_envelope(3,y_dat(start),x_dat(start),npts,
     .		           t1_date, ' ', dt_min, plot_mem_used)
	   ELSE
	      CALL PPLLDX_envelope(3,x_dat(start),y_dat(start),npts,
     .			   t1_date, ' ', dt_min, plot_mem_used)
	   ENDIF

* ... increment number of lines on plot
!	   nline_on     = nline_on     + 1
	   nline_in_mem = nline_in_mem + 1

	   npoly = npoly + 1
	GOTO 200	! end of DO-WHILE loop
 290	IF (npoly .EQ. 0 ) GOTO 5230


* ... restore normal PPLOT+ operations (else segments get prematurely created)
	   makep = ' '

* NOW set the line style (skipsym used only on line plots, dummy value here)
        do_dash = 0
	skipsym = 1
	CALL LINE_STYLE( symbol, 0.08, skipsym, line_type, line_type, use_line, 
     .                   do_dash, dashstyle, 1, 1 )
     
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
	   DO 300 ipoly = 1, npoly
 300	   x_dat(ipoly) = x_dat(1)	! single color applied to all pol'gs
	   ncolor = npoly
	ENDIF
	IF (ncolor .NE. npoly) GOTO 5220	! color array length wrong

* pass back a summary about the range of colors
	no_range = NO_LINE_RANGE(x_dat,
     .				 ncolor, mr_bad_data(mr_color), only_val )

	if (overlay) SOVER = .true.
	makep = 'FILLPOL'   ! that PLOT+ kludge again
	CALL PPLLDX_envelope(5,x_dat,x_dat,ncolor, t1_date, ' ', 
     .                       dt_min,plot_mem_used)


* If needed for the levels setting, compute data mean and std, store
* in common.  If the user set a std to use, store that.
* send 	REAL*4 rbad to compare with lev_max, lev_min inside compute_mnstd
* If the user set some /LEVELS then compute_mnstd just returns 

       rbad = bad_val4
       CALL compute_mnstd (x_dat, mr_bad_data(mr_color), need_std, 
     .    ncolor, rbad, status)
       IF (need_std .AND. status.NE.ferr_ok) THEN
* ... set up for automatic levels
          CALL PPLCMD ( from, line, 0, 'LEV,()', 1, 1 )
          CALL USE_LINEAR_LEVELS
          status = ferr_ok
       ENDIF

* If histogram-based levels are requested, compute the 
* histogram bins.

       IF (need_histo) THEN
* create temporary buffer to contain workspace

	  CALL CREATE_TEMP_MEM_VAR( cx_x, mvh_temp, status )
	  IF ( status .NE. ferr_ok ) RETURN
	  hblk1 = mr_blk1( mvh_temp )
          plot_mem_used = plot_mem_used + ncolor

	  CALL compute_histo_bins (x_dat, memory(1,hblk1), 
     .       mr_bad_data(mr_color), ncolor, status) 

* ... clean up temporary variable
	  CALL DELETE_VARIABLE( mvh_temp )
       ENDIF

* ... check for adequate PLOT+ memory to store the npts for each poly. 
* ... Save in ppl mem rather than array lleng to remove restriction on
* ... total number of polys that can be plotted.

           plot_mem_used = plot_mem_used + npoly + 1
           CALL reallo_envelope(plot_mem_used)

* reset start indices for extraction
	IF (oneD_form) THEN
	   end  = 0
	ELSE
	   xrow = cx_lo_ss(cx_x,color_dim) - 1
	   yrow = cx_lo_ss(cx_y,color_dim) - 1
	ENDIF

* loop through all the polygons getting their npts;
* loading the counts into PPLUS memory
* ---- 
         DO 700 n = 1, npoly

* ... prepare the context buffer to point to the next polygon 
	   IF ( oneD_form ) THEN
*    ... coords are 1D arrays of p1,p1,p1,p1,BAD,p2,p2,p2,p2,BAD,...
	      cx_lo_ss(cx_buff_x,coord_dim) =
     .	       cx_lo_ss(cx_buff_x,coord_dim) + end   ! advance past end of last
	      cx_lo_ss(cx_buff_y,coord_dim) =
     .	       cx_lo_ss(cx_buff_y,coord_dim) + end
	      IF (cx_lo_ss(cx_buff_x,coord_dim)
     .	     .GE. cx_hi_ss(cx_buff_x,coord_dim)) GOTO 690 ! exit DO_WHILE loop
	   ELSE
*    ... coords are 2D arrays of color_dim X coord_dim
	      xrow = xrow + 1  ! advance to next polygon in the 2D array
	      yrow = yrow + 1  ! advance to next polygon in the 2D array
	      IF (xrow .GT. cx_hi_ss(cx_x,color_dim)) GOTO 690	! exit the DO_WHILE loop
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

* ... check for missing polygon

	   missing_polygon = .false.
	   start = npts
	   end   = npts
	   DO 610 i = 1, npts
	      IF ( x_dat(i) .NE. mr_bad_data(mr_x) 
     .       .AND. y_dat(i) .NE. mr_bad_data(mr_y) ) THEN
	         start = i
	         GOTO 620
	      ENDIF
 610	   CONTINUE
	   GOTO 640   ! This polygon is all missing in X - omit		
*    ... find the end of the segment
 620	   DO 630 i = start, npts
	      IF ( x_dat(i) .EQ. mr_bad_data(mr_x) 
     .       .OR.  y_dat(i) .EQ. mr_bad_data(mr_y) ) THEN
	         end = i - 1
	         GOTO 650
	      ENDIF
 630	   CONTINUE

 640	   IF (start .GE. end ) missing_polygon = .true.

* ... check for adequate PLOT+ memory
 650	   npts = end-start+1

	   IF (missing_polygon) THEN
! for time series overlays, dont load the data
              IF (.NOT. (has_time_axis .AND. overlay)) 
     .    	 CALL pplld_pts_envelope (1, plot_mem_used)
	   ELSE
	      CALL pplld_pts_envelope (npts, plot_mem_used)
	   ENDIF

 700    CONTINUE
 690    CONTINUE
        CALL pplld_pts_envelope (ncolor, plot_mem_used)

	makep = ' '   ! that PLOT+ kludge again

	IF ( .NOT.overlay ) THEN
           CALL PPLCMD ( from, line, 0, 'XFOR 0', 1, 1 )
           CALL PPLCMD ( from, line, 0, 'YFOR 0', 1, 1 )
        ENDIF

* turn on lon/lat axis formatting, if lon/lat units, otherwise off

	IF (.NOT.overlay) THEN
	   iunits = TM_UNIT_ID(VAR_UNITS(cx_x))
	   IF (iunits .EQ. 4) THEN
	      CALL PPLCMD ( from, line, 0, 'XFOR (I7,''''LONE'''')', 1, 1 )
	   else
	      CALL PPLCMD ( from, line, 0, 'XFOR 0', 1, 1 )
	   endif
	   
	   iunits = TM_UNIT_ID(VAR_UNITS(cx_y))
	   IF (iunits .EQ. 4) THEN
	      CALL PPLCMD ( from, line, 0, 'YFOR (I7,''''LAT'''')', 1, 1 )
	   else
	      CALL PPLCMD ( from, line, 0, 'YFOR 0', 1, 1 )
	   endif
	ENDIF


* If xmodulo, check for modulo operation
* Get axis limits - if this is an overlay or the user specified /HLIM

	IF (is_modx .AND. add_lon.EQ.0) THEN
	   IF (overlay) THEN ! Get existing xaxis limits

              buffsym = 'XAXIS_MIN'
              CALL GETSYM( buffsym, buffer, nc, ier )
              READ (buffer, *) vmin
              buffsym = 'XAXIS_MAX'
              CALL GETSYM( buffsym, buffer, nc, ier )
              READ (buffer, *) vmax
	   ELSE IF ( pxdenig .GT. 0 ) THEN 
              tax = grid_line(f_dim,cx_grid(cx_x ))
              IF (tax.EQ.mnormal) tax = grid_line(t_dim,cx_grid(cx_x ))
              cal_name = line_cal_name(tax)
              cal_id = TM_GET_CALENDAR_ID ( cal_name )
	      CALL EQUAL_RANGE( 
     .			cmnd_buff(qual_start(pxdenig):qual_end(pxdenig)),
     .			x_dim, vmin, vmax, delta,
     .			formatted, range_rqd, cal_id, status )
	      IF ( status .NE. ferr_OK ) GOTO 5000
	   ENDIF

	   IF (lox.LT.vmin .OR. hix.LT.vmax) THEN
	      do_mod_plus = .TRUE.
	   ENDIF
	   IF (hix .GT. vmax .OR. lox.GT.vmin) THEN
	      do_mod_minus = .TRUE.
	   ENDIF
	   IF ((do_mod_minus.OR.do_mod_plus) .AND. x_is_log) THEN
	      CALL WARN( '/MODULO ignored on polygon command with log x axis')
	      do_mod_minus = .FALSE.
	      do_mod_plus = .FALSE.
	   ENDIF
        ENDIF

* axis scaling -- only applies if it is a fresh plot (not an overlay)
* most of the work is avoiding an error if there is no data range with
* which to make a scale ...


	IF ( .NOT.overlay ) THEN
	   delta = unspecified_val8
* ... X axis
	   IF ( pxdenig .GT. 0 ) THEN   ! explicit /XLIMITS 10/93
	      IF (pxlim.GT.0 .AND. .NOT.denig_xylim_msg_done) THEN
	         CALL WARN( '/XLIMITS and /YLIMITS are deprecated.')
	         CALL WARN( 'Use /HLIMITS and /VLIMITS instead.')
	         denig_xylim_msg_done = .TRUE.
	      ENDIF
              tax = grid_line(f_dim,cx_grid(cx_x ))
              IF (tax.EQ.mnormal) tax = grid_line(t_dim,cx_grid(cx_x ))
              cal_name = line_cal_name(tax)
              cal_id = TM_GET_CALENDAR_ID ( cal_name )

*  Check for valid log axis before proceeding if log and limits are given
	      IF (is_logx) THEN
	          CALL AXIS_ENDS(x_ax, coord_dim, cx_grid(cx_x), vmin, 
     .			     vmax, delta, x_is_log, x_axtyp, status )
	         IF ( status .NE. ferr_ok ) THEN
                    first = vmin
                    last  = vmax
                    GOTO 5400
                 ENDIF
              ENDIF

	      CALL EQUAL_RANGE( 
     .			cmnd_buff(qual_start(pxdenig):qual_end(pxdenig)),
     .			x_dim, vmin, vmax, delta,
     .			formatted, range_rqd, cal_id, status )
	      IF ( status .NE. ferr_OK ) GOTO 5000

              CALL AXIS_ENDS(x_ax, coord_dim, cx_grid(cx_x), vmin, 
     .			     vmax, delta, x_is_log, x_axtyp, status )
		 IF ( status .NE. ferr_ok ) THEN
		 first = vmin
		 last = vmax
		 GOTO 5400
	      ENDIF

	   ELSEIF (lox .GE. hix) THEN
	      IF (lox .EQ. arbitrary_large_val4) lox = 1.0
* ... no data range -- provide artifical axis limits
              CALL AXIS_ENDS(x_ax, coord_dim, cx_grid(cx_x),
     .			DBLE(lox-1.0), DBLE(lox+1.0), 0.5D0, x_is_log, 
     .			x_axtyp, status )
	      IF ( status .NE. ferr_ok ) THEN
		 first = lox-1.0
		 last = lox+1.0
		 GOTO 5400
	      ENDIF
	   ELSE
              CALL AXIS_ENDS(x_ax,coord_dim,cx_grid(cx_x),
     .	       DBLE(lox),DBLE(hix), delta, x_is_log, x_axtyp, status )
	      IF ( status .NE. ferr_ok ) THEN
		 first = lox
		 last = hix
		 GOTO 5400
	      ENDIF

	   ENDIF

* turn on lon/lat axis formatting, if lon/lat units, otherwise off

	   iunits = TM_UNIT_ID(VAR_UNITS(cx_x))
	   IF (iunits .EQ. 4) THEN
	      CALL GET_AXIS_FORMAT( DBLE(lox), DBLE(hix), delta, 
     .           fmt, use_nice )
              IF (use_nice) THEN
                  ppl_buff = 'XFOR,('//fmt(:TM_LENSTR1(fmt))//
     .              ',''''LONE'''')' 
                 CALL PPLCMD ( from, line, 0, ppl_buff, 1, 1 )
              ENDIF
           ELSE
	      CALL PPLCMD ( from, line, 0, 'XFOR 0', 1, 1 )
	   ENDIF

* ... Y axis
	   delta = unspecified_val8

	   IF ( pydenig .GT. 0 ) THEN   ! explicit /XLIMITS 10/93
	      IF (pxlim.GT.0 .AND. .NOT.denig_xylim_msg_done) THEN
	         CALL WARN( '/XLIMITS and /YLIMITS are deprecated.')
	         CALL WARN( 'Use /HLIMITS and /VLIMITS instead.')
	         denig_xylim_msg_done = .TRUE.
	      ENDIF
              tax = grid_line(f_dim,cx_grid(cx_x ))
              IF (tax.EQ.mnormal) tax = grid_line(t_dim,cx_grid(cx_x ))
              cal_name = line_cal_name(tax)
              cal_id = TM_GET_CALENDAR_ID ( cal_name )

*  Check for valid log axis before proceeding if log and limits are given
	      IF (y_is_log) THEN 
                 CALL AXIS_ENDS(y_ax,coord_dim,cx_grid(cx_y),
     .			     vmin,vmax,delta, y_is_log, y_axtyp, status )
	         IF ( status .NE. ferr_ok ) THEN
		    first = vmin
		    last = vmax
		    GOTO 5400
	         ENDIF
	      ENDIF

	      CALL EQUAL_RANGE( 
     .			cmnd_buff(qual_start(pydenig):qual_end(pydenig)),
     .			y_dim, vmin, vmax, delta,
     .			formatted, range_rqd, cal_id, status )
	      IF ( status .NE. ferr_OK ) GOTO 5000
              CALL AXIS_ENDS(y_ax,coord_dim,cx_grid(cx_y),
     .			     vmin,vmax,delta, y_is_log, y_axtyp, status )
	      IF ( status .NE. ferr_ok ) THEN
		 first = vmin
		 last = vmax
		 GOTO 5400
	      ENDIF
	   ELSEIF (loy .GE. hiy) THEN
	      IF (loy .EQ. arbitrary_large_val4) loy = 1.0
* ... no data range -- provide artifical axis limits
              CALL AXIS_ENDS(y_ax,coord_dim,cx_grid(cx_y),
     .			DBLE(loy-1.0),DBLE(loy+1.0), 0.5D0, y_is_log, 
     .			y_axtyp, status )
	      IF ( status .NE. ferr_ok ) THEN
		 first = loy-1.0
		 last = loy+1.0
		 GOTO 5400
	      ENDIF
	   ELSE
              CALL AXIS_ENDS(y_ax,coord_dim,cx_grid(cx_y),
     .		DBLE(loy),DBLE(hiy), delta, y_is_log, y_axtyp, status )
	      IF ( status .NE. ferr_ok ) THEN
		 first = loy
		 last = hiy
		 GOTO 5400
	      ENDIF

	   ENDIF

	   iunits = TM_UNIT_ID(VAR_UNITS(cx_y))
	   IF (iunits .EQ. 4) THEN
	       CALL GET_AXIS_FORMAT( DBLE(loy), DBLE(hiy), delta, 
     .            fmt, use_nice )
               IF (use_nice) THEN
                   ppl_buff = 'YFOR,('//fmt(:TM_LENSTR1(fmt))//
     .               ',''''LAT'''')' 
                  CALL PPLCMD ( from, line, 0, ppl_buff, 1, 1 )
               ENDIF
           ELSE
	      CALL PPLCMD ( from, line, 0, 'YFOR 0', 1, 1 )
	   ENDIF

*   When not an overlay, set the axis type to log or reverse log
	   IF (is_logx .OR. is_logy) THEN
	      WRITE (buff3, 3006) x_axtyp, y_axtyp

 3006         FORMAT ('axtype,', I2, ',', I2)
	      CALL PPLCMD ( from, line, 0, buff3, 1, 1)
	   ENDIF

	ELSE
	ENDIF

	IF ( no_labels ) RETURN

* * * * * * *    - - - LABELS - - -   * * * * * * * * *
* if we are starting from a blank screen each axis is labelled with the units
* of its coordinate variable
* The plot title is derived from the color value variable (unless over-ridden
* by the /TITLE qualifier) and it is placed in a key for overlays or as a
* title for a fresh plot.  If no 3rd argument, label with the word cnst as
* we are using a nominal constant as the 3rd argument.

	IF ( overlay ) THEN
	   IF ( user_title ) THEN
	      CALL EQUAL_STR_LC(
     .			cmnd_buff(qual_start(ptitl):qual_end(ptitl)),
     .                  plot_title, status )
              IF ( status .NE. ferr_ok ) GOTO 5000
	      slen = TM_LENSTR1( plot_title )
	   ELSE
	      IF (nmv .EQ. 3) THEN
	         CALL OV_GRP_CX_DOC( cx_color, 1, cx_xtra_doc )
	         plot_title = KEY_STRING( cx_color, cx_xtra_doc, slen )
	      ENDIF
	      IF (nmv .EQ. 2) THEN
	         plot_title = 'cnst'
	         slen = 4
	      ENDIF
	   ENDIF
           IF (no_range .AND. (only_val.EQ.mr_bad_data(mr_color)) ) THEN
              plot_title(slen+1:slen+14) = ' No Valid Data'
              slen = slen + 14
           ENDIF
	   CALL KEY_LOC( overlay, xkeypos, ykeypos, xkeysize, ykeysize,
     .			 lab_ht, style, justify, plot_title(:slen) )
*          Key uses line 1
	   CALL PUT_KEY( 1, plot_title(:slen),
     .			 xkeypos, ykeypos, xkeysize, ykeysize,
     .			 lab_ht, style, lab_loc_absolute, lab_loc_absolute )
	ELSE
* ... basic position and data set labels
	   CALL DISP_LABELS( cx_plot, unspecified_name4, cal_id)
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
! 5150	buff = LEFINT( INT(0.99*pplmem_nsize), slen )
!	CALL ERRMSG( ferr_prog_limit, status,
!     .		'Plot buffer size is: '//buff(:slen), *5000 )
 5160	buff = LEFINT( NPOLYGON, slen )
	CALL ERRMSG( ferr_prog_limit, status,
     .		'Maximum polygons is: '//buff(:slen), *5000 )
 5200	CALL ERRMSG( ferr_invalid_command, status,
     .		'POLYGON may use 2 args (xpts,ypts) with /LINE only,' //
     .          ' or 3 args (xpts,ypts,color_value)',
     .		*5000)
 5210	CALL ERRMSG( ferr_invalid_command, status,
     .		'illegal dimensions for polygon coordinate(s) or values',
     .		*5000)
 5220	buff2 = LEFINT(npoly, slen2)
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

 5360	cal_name = TM_GET_CALENDAR_NAME(cal_id)
        cal_name_saved = TM_GET_CALENDAR_NAME(saved_calendar_id)
        slen = TM_LENSTR1 (cal_name_saved)
        slen2 = TM_LENSTR1(cal_name)
	CALL ERRMSG( ferr_inconsist_grid, status,
     .          'Differing calendar axes: '//pCR//
     .		'first variable is on '//cal_name_saved(:slen)//
     .		' axis'//pCR//
     .		'overlaid variable is on '//cal_name(:slen2)//
     .		' axis', *5000 )

 5400	buff3 = LEFT_REAL (first, '(G15.3)', slen)
	buff4 = LEFT_REAL (last,  '(G15.3)', slen2)
	CALL ERRMSG( ferr_out_of_range, status,
     . 'Limits for log axis negative or too small: '//
     .  buff3(:slen)// ' : '//  buff4(:slen2), *5000 )

 5600	buff3 = LEFT_REAL (small, '(G15.3)', slen)
	CALL ERRMSG( ferr_out_of_range, status,
     . 'X coordinates on log axis negative or too small : '//
     .  buff3(:slen), *5000 )
	END
