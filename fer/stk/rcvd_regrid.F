	SUBROUTINE RCVD_REGRID( src_cx, dst_cx, axis, work, ok_min, ok_max )

* given the reveived ("rcvd") source context (of actually available data) and
* the requested output region determine the limits of the result that can be
* calculated

* programmer - steve hankin
* NOAA/PMEL, Seattle, WA - Tropical Modeling and Analysis Program
* written for VAX computer under VMS operating system

* V200:  12/1/89 - extracted from IS_REGRID
* 	 2/27/90 - if destination is unspecified set it from the source
* V230:  8/22/92 - special bypass for regridding by association
* V300:   5/5/93 - special check for source grid with only one point
*	         - and use single precision check comparing world coords
*	NOTE: sing. prec. check is risky - consistent w/ regridding algorithms?
* V402:	 8/16/95 - bug fix: T axis was **assumed** to be calendar
* V420:	   10/95 - determine regriddable limits for "window" transforms
*			such as min, max, ...
*	   11/95 - modulo regridding - source region independent of result
*	 1/96    - shifted @SUM and @NGD to weighted calcs like @AVE
* V490:9/97 *kob* - mods to include negative time axis processing
*   10/6/97  *sh* - added g=@xact regridding 
* V491:4/98  *sh* - bug fix: @ASN could lead to hi<lo limits

* calling argument declarations:
	INTEGER	src_cx, dst_cx, axis, work, ok_min, ok_max

* internal variable declarations:
	LOGICAL GEOG_LABEL, ITSA_MODULO_REGRID, calendar, modulo
	INTEGER ISUBSCRIPT, CAXIS_LEN,
     .		where_loest, where_hiest, rgrd_trans,
     .		dst_ss_min, dst_ss_max,
     .		dst_grid, src_grid,
     .		src_lo_ss, src_hi_ss, dst_lo_ss, dst_hi_ss
	REAL*8	TWORLD, TM_WORLD, SECS_TO_TSTEP,
     .		src_lo_ww, src_hi_ww, dst_lo_ww, dst_hi_ww,
     .          tmp_lo_ww, tmp_hi_ww

#ifdef unix
	include	'ferret_cmn/ferret.parm'
	include	'ferret_cmn/xcontext.cmn'
#else
	INCLUDE	'FERRET_CMN:FERRET.PARM'
	INCLUDE	'FERRET_CMN:XCONTEXT.CMN'
#endif

* initialize
	src_grid = cx_grid( src_cx )
	dst_grid = cx_grid( dst_cx )
* ... regrid calendar time axes? (src_grid was checked in rqst_regrid)
	calendar = axis.EQ.t_dim .AND. GEOG_LABEL(axis,dst_grid)
	modulo = ITSA_MODULO_REGRID(dst_cx,axis)

* what is the regridding transformation for this axis ?
	rgrd_trans = cx_regrid_trans( axis, dst_cx )

* easy bypass for regridding by association
        IF ( rgrd_trans .EQ. prgrd_associate ) THEN
           CALL GRID_SUBSCRIPT_EXTREMES(dst_ss_min, dst_ss_max,
     .                                  dst_grid, axis)
           dst_lo_ss = MAX( cx_lo_ss(src_cx,axis), dst_ss_min )
           dst_hi_ss = MIN( cx_hi_ss(src_cx,axis), dst_ss_max )
	   IF (dst_lo_ss .GT. dst_hi_ss) dst_hi_ss = dst_lo_ss	! 4/98
           GOTO 500

* ... and easy bypass for regridding by modulo
* ... set default limits at 1 to n regardless
* ... (cant use GRID_SS_EXTR because of line_modulo)
        ELSEIF ( modulo ) THEN
           dst_lo_ss = 1
!           dst_hi_ss = line_dim(grid_line(axis,dst_grid)) ! needs TMAP CMN
           dst_hi_ss = CAXIS_LEN(axis, dst_cx)
           GOTO 500
        ENDIF

* for interpolation we need source points surrounding the destination points
* for averaging     we need source area   surrounding the destination area
* for "window"ing   we need source area   containing  the destination area
* for exact match   we need source points aligning w/ the destination points
	IF ( rgrd_trans .EQ. prgrd_linear_interp ) THEN
	   where_loest = box_middle
	   where_hiest = box_middle
	ELSEIF ( rgrd_trans.GE.prgrd_min
     .	   .OR.  rgrd_trans.EQ.prgrd_max ) THEN
	   where_loest = box_lo_lim			! unweighted
	   where_hiest = box_hi_lim
	ELSEIF ( rgrd_trans .EQ. prgrd_xy_ave
     .      .OR. rgrd_trans .EQ. prgrd_variance
     .      .OR. rgrd_trans .EQ. prgrd_sum
     .      .OR. rgrd_trans .EQ. prgrd_good_pt
     .	    .OR. rgrd_trans .EQ. prgrd_average ) THEN
	   where_loest = box_lo_lim
	   where_hiest = box_hi_lim
	ELSEIF ( rgrd_trans .EQ. prgrd_xact_pts ) THEN	! 10/97
	   where_loest = box_middle
	   where_hiest = box_middle
	ENDIF

* find the available source region for regridding
* NOTE:  times expressed as seconds since BC
	src_lo_ww = TWORLD(cx_lo_ss(src_cx,axis),src_grid,axis,where_loest)
	src_hi_ww = TWORLD(cx_hi_ss(src_cx,axis),src_grid,axis,where_hiest)

* find the corresponding subscripts on the destination grid
* 9/97 *kob* convert time in secs since BC to time position on the 
*             *source* time axis then call ISUBSCRIPT
	IF ( calendar ) THEN	! fixed 8/95
	   tmp_lo_ww = SECS_TO_TSTEP(dst_grid, src_lo_ww)
	   tmp_hi_ww = SECS_TO_TSTEP(dst_grid, src_hi_ww)
	   dst_lo_ss = ISUBSCRIPT( tmp_lo_ww,dst_grid,axis,round_up )
	   dst_hi_ss = ISUBSCRIPT( tmp_hi_ww,dst_grid,axis,round_dn )
	ELSE
	   dst_lo_ss = ISUBSCRIPT( src_lo_ww,dst_grid,axis,round_up )
	   dst_hi_ss = ISUBSCRIPT( src_hi_ww,dst_grid,axis,round_dn )
	ENDIF


* determine whether destination edges can actually be regridded
* changed to sing. prec 5/93
	dst_lo_ww = TWORLD(dst_lo_ss,dst_grid,axis,where_loest)
	dst_hi_ww = TWORLD(dst_hi_ss,dst_grid,axis,where_hiest)
	IF ( SNGL(dst_lo_ww) .LT. SNGL(src_lo_ww) )
     .	                                dst_lo_ss = dst_lo_ss + 1
	IF ( SNGL(dst_hi_ww) .GT. SNGL(src_hi_ww) )
     .                                  dst_hi_ss = dst_hi_ss - 1
	IF ( dst_lo_ss .GT. dst_hi_ss ) dst_hi_ss = dst_lo_ss  ! bug check 5/93

* if destination limits were unspecified set them equal to the available data
 500    IF ( cx_hi_ss(dst_cx,axis) .EQ. unspecified_int4 ) THEN
	   cx_lo_ss(dst_cx,axis) = dst_lo_ss
	   cx_hi_ss(dst_cx,axis) = dst_hi_ss
	   cx_lo_ww(axis,dst_cx) = TM_WORLD( dst_lo_ss,
     .					     dst_grid, axis, box_lo_lim )
	   cx_hi_ww(axis,dst_cx) = TM_WORLD( dst_hi_ss,
     .					     dst_grid, axis, box_hi_lim )
	ENDIF

* even with sufficient source data we shouldn't exceed the requested region
        dst_ss_min = cx_lo_ss(dst_cx,axis)
	dst_ss_max = cx_hi_ss(dst_cx,axis)

* determine the limits that can actually be regridded
	IF ( modulo ) THEN
           ok_min = dst_ss_min		! can always regrid to full request
	   ok_max = dst_ss_max
	ELSE
           ok_min = MAX( dst_lo_ss, dst_ss_min )
	   ok_max = MIN( dst_hi_ss, dst_ss_max )
	ENDIF

* size of working arrays needed for later regridding
	work = dst_hi_ss - dst_lo_ss + 2

	RETURN
	END	
