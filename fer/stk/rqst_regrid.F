	SUBROUTINE RQST_REGRID( dst_cx, src_cx, axis, strides, status )

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


* given a destination context and an axis to be regridded
* determine the source data limits that should be requested ("rqst")

* programmer - steve hankin
* NOAA/PMEL, Seattle, WA - Tropical Modeling and Analysis Program

* V200:  12/1/89 - extracted from REGRID_LIMS
* 	 2/26/90 - repaired minor bug in error message
* 	  3/2/90 - changed inequality conditions to allow endpoints exactly =
* V230:  8/22/92 - allow regridding by association and by single axis averaging
*        11/4/92 - expanded variable name length
* V300:  3/16/93 - if destination region doesn't overlap the source region
*                  then request a single point at the closest edge of the src
*                  This allows bad data to be returned instead of an error.
* V402:	 8/16/95 - bug fix: T axis was **assumed** to be calendar
* V420: 10/25/95 - do not let subscript limits (src_lo/hi_ss) exceed bounds
*	11/95	 - request unspec limits for modulo regrid (entire src region)
*		   (1/96 - set cx_given to FALSE ... omitted earlier)
*	 1/96    - shifted @SUM and @NGD to weighted calcs like @AVE
*	 2/96    - issue warning that G=@NRS is not implemented
* V490:  9/97 *kob* - mods to include negative time axis processing
*     10/6/97 *sh*  - added g=@xact regridding 
* V500:  2/99 *sh* - modified behavior when region is unspecified -- request
*		full  dest axis limits of source rather than unspecified limts
*  *kob* 3/99- up VAR_CODE  to 64 chars 
*        5/99 *sh* for netCDF vars of form var[I=1:10:3] pass along the
*		  delta value in cx_delta to be used as a "stride" in I/O 
* V530   3/01 *acm* alternative calendar axes:  Compare timesteps for source and
*		    destination axes with respect to one of the calendars.
* V530:  3/01 *sh* - added prgrd_nearest

#ifdef unix
	include 'tmap_dims.parm'
	include	'xtm_grid.cmn_text'
	external xgt_grid_data
	include	'ferret.parm'
	include	'errmsg.parm'
	include	'xcontext.cmn'
	include 'xrisc.cmn'
#else
	INCLUDE 'TMAP_FORMAT:TMAP_DIMS.PARM'
	INCLUDE	'TMAP_FORMAT:XTM_GRID.CMN'
	INCLUDE	'FERRET_CMN:FERRET.PARM'
	INCLUDE	'FERRET_CMN:ERRMSG.PARM'
	INCLUDE	'FERRET_CMN:XCONTEXT.CMN'
#endif
!	INCLUDE	'FERRET_CMN:XVARIABLES.CMN'
#ifdef unix
	include	'xalgebra.cmn'
	include	'xtext_info.cmn'
#else
	INCLUDE	'FERRET_CMN:XALGEBRA.CMN'
	INCLUDE	'FERRET_CMN:XTEXT_INFO.CMN'
#endif

* calling argument declarations:
	LOGICAL strides
	INTEGER	src_cx, dst_cx, status

* V500 *kob* 3/99- up VAR_CODE  to 64 chars 
* internal variable declarations:
	LOGICAL	GEOG_LABEL, ITSA_MODULO_REGRID,
     .		last_axis, src_modulo, calendar
	INTEGER	ISUBSCRIPT, CAXIS_LEN, axis, TM_GET_CALENDAR_ID, TM_LENSTR1,
     .		dst_grid, src_grid, src_lo_ss, src_hi_ss,
     .		src_min_ss, src_max_ss, line, src_line,
     .		dst_lo_ss, dst_hi_ss, where_loest, where_hiest, rgrd_trans,
     .	        dst_taxis, src_taxis, dst_cal_id, src_cal_id, dlen, slen
	REAL*8	TWORLD, SECS_TO_TSTEP, TSTEP_TO_SECS,
     .		src_lo_ww, src_hi_ww, dst_lo_ww, dst_hi_ww,
     .          tmp_lo_ww, tmp_hi_ww, d2s_lo_ww, d2s_hi_ww
	CHARACTER	VAR_CODE*6, TM_SECS_TO_DATE*20, pdate*20

* initialize
	dst_grid = cx_grid( dst_cx )
	src_grid = cx_grid( src_cx )

* get regrid transform.  remove its flag from the source request
	rgrd_trans = cx_regrid_trans( axis, dst_cx )
	cx_regrid_trans(axis,src_cx) = prgrd_exact

* ! temp kludge - allow regridding to higher dimensioned objects
	IF ( grid_line(axis,src_grid) .EQ. mnormal ) GOTO 1000

* determine if this is a calendar axis and trap illegal attempts
* to regrid between calendar and non-calendar axes.  
* 3/01 *acm* If calendar axis, get calendar id for dest and source calendars.

	IF ( rgrd_trans.NE.prgrd_associate .AND. axis.EQ.t_dim ) THEN
	   calendar = GEOG_LABEL(t_dim,dst_grid)
	   IF (GEOG_LABEL(t_dim,src_grid).NEQV.calendar) GOTO 5010
	   IF (calendar) THEN
	      src_taxis = grid_line(axis, src_grid)
	      src_cal_id = TM_GET_CALENDAR_ID (line_cal_name(src_taxis) )
	      dst_taxis = grid_line(axis, dst_grid)
	      dst_cal_id = TM_GET_CALENDAR_ID (line_cal_name(dst_taxis) )
	   ENDIF

	ELSE
	   calendar = .FALSE.
	ENDIF

* If this is a regridding between different calendars trap illegal attempts
* to regrid with anything except @lin, @asn, or @near
	IF ( calendar ) THEN
	   dst_taxis = grid_line(t_dim, dst_grid )
	   src_taxis = grid_line(t_dim, src_grid )

	   dst_cal_id = TM_GET_CALENDAR_ID (line_cal_name(dst_taxis))
	   src_cal_id = TM_GET_CALENDAR_ID (line_cal_name(src_taxis))
           IF (dst_cal_id .NE. src_cal_id) THEN
               IF (rgrd_trans .NE. prgrd_linear_interp .AND. 
     .		   rgrd_trans .NE. prgrd_associate     .AND. 
     .		   rgrd_trans .NE. prgrd_nearest )  GOTO 5020
           ENDIF
	ELSE
	   calendar = .FALSE.
	ENDIF

* modulo regridding always requests the entire source region
	IF ( ITSA_MODULO_REGRID(dst_cx,axis) ) GOTO 1000

* for interpolation we need source points surrounding the destination points
* for averaging     we need source area   surrounding the destination area
* for exact match   we need source points aligning w/ the destination points
	IF ( rgrd_trans .EQ. prgrd_linear_interp ) THEN
	   where_loest = box_middle
	   where_hiest = box_middle
	ELSEIF ( rgrd_trans.EQ.prgrd_min
     .	   .OR.  rgrd_trans.EQ.prgrd_max ) THEN
	   where_loest = box_lo_lim			! unweighted
	   where_hiest = box_hi_lim
	ELSEIF ( rgrd_trans .EQ. prgrd_xy_ave 
     .      .OR. rgrd_trans .EQ. prgrd_variance
     .      .OR. rgrd_trans .EQ. prgrd_sum
     .      .OR. rgrd_trans .EQ. prgrd_good_pt
     .      .OR. rgrd_trans .EQ. prgrd_average ) THEN
	   where_loest = box_lo_lim
	   where_hiest = box_hi_lim
	ELSEIF ( rgrd_trans .EQ. prgrd_associate ) THEN
           CONTINUE
	ELSEIF ( rgrd_trans .EQ. prgrd_xact_pts
     .	    .OR. rgrd_trans .EQ. prgrd_nearest ) THEN
	   where_loest = box_middle
	   where_hiest = box_middle
	ELSE
	   CALL ERRMSG( ferr_internal, status, 'rgrd transf', *5000 )
	ENDIF

* locate the necessary destination region (time in secs since BC)
	IF ( cx_lo_ss(dst_cx,axis) .EQ. unspecified_int4 ) THEN
! 2/99 - commented out earlier versions of logic
!* ... if unknown, hope they will get figured out later <-- current on 2/99
!	   GOTO 1000
!* ... error (for now) if limits are unspecified	<-- old on 2/99
!!	   GOTO 5200
* ... always try to fill the entire destination axis
	   dst_lo_ss = 1
	   dst_hi_ss = CAXIS_LEN(axis, dst_cx)
	ELSE
	   dst_lo_ss = cx_lo_ss(dst_cx,axis)
	   dst_hi_ss = cx_hi_ss(dst_cx,axis)
	ENDIF

	CALL GRID_SUBSCRIPT_EXTREMES(src_min_ss, src_max_ss, src_grid, axis)
        IF ( rgrd_trans .EQ. prgrd_associate ) THEN     ! 8/92
* ... regridding by association bypasses all the hard stuff
	   src_lo_ss = src_min_ss
	   src_hi_ss = src_max_ss
           IF     ( dst_lo_ss .GT. src_hi_ss ) THEN   ! 3/93
              src_lo_ss = src_hi_ss                   ! 3/93 request top point
           ELSEIF ( dst_hi_ss .LT. src_lo_ss ) THEN   ! 3/93
              src_hi_ss = src_lo_ss        ! 3/93 request low point (possible?)
           ELSE
	      IF ( dst_lo_ss .GT. src_lo_ss ) src_lo_ss = dst_lo_ss
	      IF ( dst_hi_ss .LT. src_hi_ss ) src_hi_ss = dst_hi_ss
           ENDIF
           GOTO 500
        ELSE
* NOTE:  times expressed as seconds since BC
           dst_lo_ww = TWORLD( dst_lo_ss,dst_grid,axis,where_loest ) 
	   dst_hi_ww = TWORLD( dst_hi_ss,dst_grid,axis,where_hiest )
        ENDIF

* check to make sure destination region overlaps the source region
* (temporarily use src_**_ww for axis extremes)
* 3/01 *acm* If different calendars, do comparison wrt source calendar...

	src_modulo = line_modulo( grid_line(axis,src_grid) )
	IF ( .NOT. src_modulo ) THEN
	   CALL GRID_SUBSCRIPT_EXTREMES(src_lo_ss, src_hi_ss, src_grid, axis)
	   src_lo_ww = TWORLD( src_lo_ss,src_grid,axis,where_loest )
	   src_hi_ww = TWORLD( src_hi_ss,src_grid,axis,where_hiest )

! (3/2)    IF ( dst_lo_ww.GE.src_hi_ww .OR. dst_hi_ww.LE.src_lo_ww ) GOTO 5100
! (3/93)   IF ( dst_lo_ww.GT.src_hi_ww .OR. dst_hi_ww.LT.src_lo_ww ) GOTO 5100

           IF (calendar .AND. src_cal_id .NE. dst_cal_id) THEN
              CALL reset_secs_from_bc (dst_lo_ww, tmp_lo_ww, 
     .                                 dst_cal_id, src_cal_id, status)
	      IF ( status .NE. ferr_ok ) GOTO 5000
	      CALL reset_secs_from_bc (dst_hi_ww, tmp_hi_ww, 
     .                                 dst_cal_id, src_cal_id, status)
	      IF ( status .NE. ferr_ok ) GOTO 5000
	   ELSE
	      tmp_lo_ww = dst_lo_ww
	      tmp_hi_ww = dst_hi_ww
	   ENDIF

           IF     ( tmp_lo_ww .GT. src_hi_ww ) THEN   ! 3/93
              src_lo_ss = src_hi_ss                   ! 3/93 request top point
              GOTO 500                                ! 3/93
           ELSEIF ( tmp_hi_ww .LT. src_lo_ww ) THEN   ! 3/93
              src_hi_ss = src_lo_ss                   ! 3/93 request low point
              GOTO 500                                ! 3/93
           ENDIF
	ENDIF

* which source grid boxes contain the destination region
* 	(not necessarily adequate for regridding)
!	IF ( axis .EQ. t_dim ) THEN	! pre-8/95
* 9/97 *kob* convert time in secs since BC to time position on the 
*             *source* time axis then call ISUBSCRIPT
	IF ( calendar ) THEN

* 3/01 *acm* Get timesteps for dst range wrt source calendar
	   IF (dst_cal_id .NE. src_cal_id) THEN
              CALL reset_secs_from_bc (dst_lo_ww, d2s_lo_ww, 
     .                                 dst_cal_id, src_cal_id, status)
	      IF ( status .NE. ferr_ok ) GOTO 5000
              CALL reset_secs_from_bc (dst_hi_ww, d2s_hi_ww, 
     .                                 dst_cal_id, src_cal_id, status)
	      IF ( status .NE. ferr_ok ) GOTO 5000
	   ELSE
	      d2s_lo_ww = dst_lo_ww
	      d2s_hi_ww = dst_hi_ww
	   ENDIF

	   tmp_lo_ww = SECS_TO_TSTEP( src_grid, d2s_lo_ww)
	   tmp_hi_ww = SECS_TO_TSTEP( src_grid, d2s_hi_ww)

	   src_lo_ss = ISUBSCRIPT( tmp_lo_ww,src_grid,axis,round_dn )
	   src_hi_ss = ISUBSCRIPT( tmp_hi_ww,src_grid,axis,round_up )
	ELSE
	   src_lo_ss = ISUBSCRIPT( dst_lo_ww,src_grid,axis,round_dn )
	   src_hi_ss = ISUBSCRIPT( dst_hi_ww,src_grid,axis,round_up )
	ENDIF
* 9/97 *kob* now convert to ss values


* determine the maximum regriddable region from this source data
	src_lo_ww = TWORLD( src_lo_ss, src_grid, axis, where_loest )
	src_hi_ww = TWORLD( src_hi_ss, src_grid, axis, where_hiest )

* if this isn't sufficient, expand it
* 10/95:  Avoid expanding beyond the axis limits.  This becomes necessary
*  despite the call to CONFINE_AXIS because dynamic grids contexts are immune
*  from CONFINE_CONTEXT as a result of uvar_given = uvlim_gvn_gnl

* 3/01 *acm* If different calendars, do comparison wrt dst calendar...

        IF (calendar .AND. dst_cal_id .NE. src_cal_id) THEN
	   tmp_lo_ww = d2s_lo_ww
	   tmp_hi_ww = d2s_hi_ww
	ELSE
	   tmp_lo_ww = dst_lo_ww
	   tmp_hi_ww = dst_hi_ww
	ENDIF

	IF ( tmp_lo_ww .LT. src_lo_ww
     . .AND. src_lo_ss .GT. src_min_ss ) src_lo_ss = src_lo_ss - 1
	IF ( tmp_hi_ww .GT. src_hi_ww
     . .AND. src_hi_ss .LT. src_max_ss ) src_hi_ss = src_hi_ss + 1

* flesh out the resulting axis
 500    cx_by_ss( axis, src_cx ) = .TRUE.
	cx_lo_ss( src_cx, axis ) = src_lo_ss
	cx_hi_ss( src_cx, axis ) = src_hi_ss
	CALL CONFINE_AXIS( axis, src_cx, status )
	IF ( status .NE. ferr_ok ) GOTO 5000
	CALL FLESH_OUT_AXIS( axis, src_cx, status )
	IF ( status .NE. ferr_ok ) GOTO 5000

* (5/99) flag the use of netCDF strides by inserting cx_delta value
	IF (strides) THEN
	   line = grid_line(axis,dst_grid)
	   IF (line_regular(line)) THEN
	      src_line  = grid_line(axis,src_grid)
	      cx_delta(axis,src_cx) = line_delta(line)
     .				    / line_delta(src_line)
	   ELSE
	      cx_delta(axis,src_cx) = line_delta(line)
	   ENDIF
	ENDIF

* success
	RETURN

* set axis region as still unknown at this stage
 1000	cx_lo_ss( src_cx, axis )	= unspecified_int4
	cx_hi_ss( src_cx, axis )	= unspecified_int4
	cx_lo_ww( axis, src_cx )	= unspecified_val8
	cx_hi_ww( axis, src_cx )	= unspecified_val8
	cx_given( axis, src_cx )	= .FALSE.
* 9/97 *kob* - cx_calendar needs to also reflect these changes
	IF (axis .eq. t_dim)
     .      cx_calendar(src_cx) = .FALSE.

	RETURN

* error exit
 5000	RETURN
 5010	CALL ERRMSG( ferr_regrid, status,
     .		'only @ASN regridding between calendar and '//pCR
     .		//'non-calendar axes: '//
     .		VAR_CODE(cx_category(dst_cx),cx_variable(dst_cx)), *5000)
 5020	CONTINUE

	slen = TM_LENSTR1 (line_cal_name(src_taxis))
	dlen = TM_LENSTR1 (line_cal_name(dst_taxis))
	WRITE (risc_buff, *) 'only @ASN, @LIN, or @NRST regridding ',
     .		'between calendar types: ',
     .		line_cal_name(src_taxis)(:slen),', ', 
     .		line_cal_name(dst_taxis)(:dlen)
	CALL ERRMSG( ferr_regrid, status, risc_buff, *5000)
! 5100	CALL ERRMSG( ferr_limits, status,
!     .		'regrid region outside source data '//ww_dim_name(axis)//
!     .		' axis range: '//
!     .		VAR_CODE(cx_category(dst_cx),cx_variable(dst_cx)), *5000)
! 5200	CALL ERRMSG( ferr_limits, status,
!     .		'regrid requires limits given on '//ww_dim_name(axis)//
!     .		' axis of '//
!     .		VAR_CODE(cx_category(dst_cx),cx_variable(dst_cx)), *5000)
	END
