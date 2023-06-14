	SUBROUTINE RCVD_REGRID( src_cx, dst_cx, axis, work, 
     .				ok_min, ok_max, status )

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
* V500: 2/99 *sh* - modified behavior when dest region is unspecified -- 
*		provide full dest axis limits rather than source-data driven
* V510: 1/00 *sh* - bug fix: destination subscript limits chosen incorrectly
*		for min, max, ave, etc.
*		  - bigger work array for DO_WINDOW_REGRID
* V530   3/01 *acm* alternative calendar axes:  Compare timesteps for source and
*		    destination axes with respect to one of the calendars.
* V530: 3/01 *sh* - added rgrd_nearest -- (unrestricted by source lims)
* V550: 2/02 *sh* - did NOT change CAXIS_LEN to CAXIS_MODLEN
*		    since subspan modulo default length is N not N+1
* v580 *acm* 11/04 Store cal_id in cx_cal_id, the calendar ID used when  
*                  computing world coordinates
* V671  6/11 *acm*  New 2D linear xy regridding, fixing ticket 1862
*       *acm* 3/12 cleanup ifdefs and unnecessary include files
*       *acm* 4/12 6D Ferret: time axis may be in t or f direction. The time 
*                  dimension is now an arg to SECS_TO_TSTEP. Note the input 
*                  argument "axis" in this routine is a dimension, 1, 2, ... 
* V696 *acm* 11/15 Added @BIN and @NBIN regridding transforms
* V720: 6/17 *acm* Ticket 2346: new @IIN regridding transform

        IMPLICIT NONE
* calling argument declarations:
	INTEGER	src_cx, dst_cx, axis, work, ok_min, ok_max, status

* internal variable declarations:
	LOGICAL GEOG_LABEL, ITSA_MODULO_REGRID, calendar, modulo,
     .		need_margin_check
	INTEGER ISUBSCRIPT, CAXIS_LEN, TM_GET_CALENDAR_ID,
     .		dst_where_loest, dst_where_hiest, rgrd_trans,
     .		src_where_loest, src_where_hiest,
     .		dst_ss_min, dst_ss_max,
     .		dst_grid, src_grid,
     .		dst_lo_ss, dst_hi_ss,
     .          src_taxis, dst_taxis, src_cal_id, dst_cal_id
	REAL*8	TWORLD, TM_WORLD, SECS_TO_TSTEP,
     .		src_lo_ww, src_hi_ww, dst_lo_ww, dst_hi_ww,
     .          tmp_lo_ww, tmp_hi_ww, s2d_lo_ww, s2d_hi_ww

	include 'tmap_dims.parm'
	include	'xtm_grid.cmn_text'
	external xgt_grid_data
	include	'ferret.parm'
	include	'errmsg.parm'
	include	'xcontext.cmn'

* initialize
	status = ferr_ok
	src_grid = cx_grid( src_cx )
	dst_grid = cx_grid( dst_cx )
* ... regrid calendar time axes? (src_grid was checked in rqst_regrid)
	calendar = (axis.EQ.t_dim .OR. axis.EQ.f_dim) .AND. GEOG_LABEL(axis,dst_grid)
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
           dst_hi_ss = CAXIS_LEN(axis, dst_cx)
           GOTO 500
        ENDIF

* for interpolation we need source points surrounding the destination points
* for averaging     we need source area   surrounding the destination area
* for "window"ing   we need source area   containing  the destination area
* for exact match   we need source points aligning w/ the destination points
* for nrst match    do as for averaging
* for binning       do as for averaging
	IF ( rgrd_trans .EQ. prgrd_linear_interp
     .      .OR. rgrd_trans .EQ. prgrd_xy_lin ) THEN
	   dst_where_loest = box_middle
	   dst_where_hiest = box_middle
	   src_where_loest = box_middle
	   src_where_hiest = box_middle
	   need_margin_check = .TRUE.
	ELSEIF ( rgrd_trans .EQ. prgrd_min
     .	    .OR. rgrd_trans .EQ. prgrd_max ) THEN
	   dst_where_loest = box_lo_lim
	   dst_where_hiest = box_hi_lim
	   src_where_loest = box_middle
	   src_where_hiest = box_middle
	   need_margin_check = .FALSE.
	ELSEIF ( rgrd_trans .EQ. prgrd_xy_ave
     .      .OR. rgrd_trans .EQ. prgrd_variance
     .      .OR. rgrd_trans .EQ. prgrd_sum
     .      .OR. rgrd_trans .EQ. prgrd_good_pt
     .	    .OR. rgrd_trans .EQ. prgrd_average
     .	    .OR. rgrd_trans .EQ. prgrd_nearest
     .	    .OR. rgrd_trans .EQ. prgrd_bin
     .	    .OR. rgrd_trans .EQ. prgrd_nbin
     .	    .OR. rgrd_trans .EQ. prgrd_iin ) THEN   ! I think @iin is whole box
	   dst_where_loest = box_lo_lim
	   dst_where_hiest = box_hi_lim
	   src_where_loest = box_lo_lim
	   src_where_hiest = box_hi_lim
	   need_margin_check = .FALSE.
	ELSEIF ( rgrd_trans .EQ. prgrd_xact_pts
     .	    .OR. rgrd_trans .EQ. prgrd_nearest ) THEN
	   dst_where_loest = box_middle
	   dst_where_hiest = box_middle
	   src_where_loest = box_middle
	   src_where_hiest = box_middle
	   need_margin_check = .TRUE.
	ENDIF

* find the available source region for regridding
* NOTE:  times expressed as seconds since BC
	src_lo_ww = TWORLD(cx_lo_ss(src_cx,axis),src_grid,axis,
     .			   src_where_loest)
	src_hi_ww = TWORLD(cx_hi_ss(src_cx,axis),src_grid,axis,
     .			   src_where_hiest)

* find the corresponding subscripts on the destination grid
* 9/97 *kob* convert time in secs since BC to time position on the 
*             *source* time axis then call ISUBSCRIPT
	IF ( calendar ) THEN	! fixed 8/95

* 3/01 *acm* get timestep for source range relative to destination calendar.

	   src_taxis = grid_line(axis, src_grid)
	   src_cal_id = TM_GET_CALENDAR_ID (line_cal_name(src_taxis) )

	   dst_taxis = grid_line(axis, dst_grid)
	   dst_cal_id = TM_GET_CALENDAR_ID (line_cal_name(dst_taxis) )

	   IF (src_cal_id .NE. dst_cal_id) THEN
	      CALL reset_secs_from_bc (src_lo_ww, s2d_lo_ww, 
     .                                 src_cal_id, dst_cal_id, status)
	      IF ( status .NE. ferr_ok ) GOTO 5000
	      CALL reset_secs_from_bc (src_hi_ww, s2d_hi_ww, 
     .                                 src_cal_id, dst_cal_id, status)
	      IF ( status .NE. ferr_ok ) GOTO 5000
	   ELSE
              s2d_lo_ww = src_lo_ww
              s2d_hi_ww = src_hi_ww
	   ENDIF

	   tmp_lo_ww = SECS_TO_TSTEP(dst_grid, axis, s2d_lo_ww)
	   tmp_hi_ww = SECS_TO_TSTEP(dst_grid, axis, s2d_hi_ww)

	   dst_lo_ss = ISUBSCRIPT( tmp_lo_ww,dst_grid,axis,round_up )
	   dst_hi_ss = ISUBSCRIPT( tmp_hi_ww,dst_grid,axis,round_dn )
	ELSE
	   dst_lo_ss = ISUBSCRIPT( src_lo_ww,dst_grid,axis,round_up )
	   dst_hi_ss = ISUBSCRIPT( src_hi_ww,dst_grid,axis,round_dn )
	ENDIF


* determine whether destination edges can actually be regridded
* changed to sing. prec 5/93
	dst_lo_ww = TWORLD(dst_lo_ss,dst_grid,axis,dst_where_loest)
	dst_hi_ww = TWORLD(dst_hi_ss,dst_grid,axis,dst_where_hiest)
! 1/00 removed this check for G=@MIN, MAX, ...
!      IS IT NEEDED AT ALL???
	IF ( need_margin_check ) THEN

* 3/01 *acm* If different calendars, do comparison wrt dst calendar...

           IF (calendar .AND. src_cal_id .NE. dst_cal_id) THEN
	      CALL reset_secs_from_bc (src_lo_ww, tmp_lo_ww, 
     .                                 src_cal_id, dst_cal_id, status)
	      IF ( status .NE. ferr_ok ) GOTO 5000
	      CALL reset_secs_from_bc (src_hi_ww, tmp_hi_ww, 
     .                                 src_cal_id, dst_cal_id, status)
	      IF ( status .NE. ferr_ok ) GOTO 5000

	   ELSE
	      tmp_lo_ww = src_lo_ww
	      tmp_hi_ww = src_hi_ww
	   ENDIF
	   IF ( (dst_lo_ww) .LT. (tmp_lo_ww) )
     .	                                dst_lo_ss = dst_lo_ss + 1
	   IF ( (dst_hi_ww) .GT. (tmp_hi_ww) )
     .                                  dst_hi_ss = dst_hi_ss - 1

	ENDIF
	IF ( dst_lo_ss .GT. dst_hi_ss ) dst_hi_ss = dst_lo_ss  !bug chk 5/93

* if destination limits were unspecified set them to the full axis span
 500    IF ( cx_hi_ss(dst_cx,axis) .EQ. unspecified_int4 ) THEN
! pre 2/99 logic for unspecified limits
!	   cx_lo_ss(dst_cx,axis) = dst_lo_ss
!	   cx_hi_ss(dst_cx,axis) = dst_hi_ss
	   cx_lo_ss(dst_cx,axis) = 1
	   cx_hi_ss(dst_cx,axis) = CAXIS_LEN(axis, dst_cx)
	   cx_lo_ww(axis,dst_cx) = TM_WORLD( dst_lo_ss,
     .					     dst_grid, axis, box_lo_lim )
	   cx_hi_ww(axis,dst_cx) = TM_WORLD( dst_hi_ss,
     .					     dst_grid, axis, box_hi_lim )
           cx_cal_id(dst_cx) = dst_cal_id
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
* 1/00 times two to accomodate @MIN, @MAX needs
	work = ( dst_hi_ss - dst_lo_ss + 2 ) * 2
	RETURN
* error exit
 5000	RETURN
	END	
