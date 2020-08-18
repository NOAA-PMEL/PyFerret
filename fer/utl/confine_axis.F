	SUBROUTINE CONFINE_AXIS( idim, cx, status )

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
* given a skeletal context with a known grid confine the context limits to
* on the given axis within the grid limits
* if date/time limits were specified convert to time steps here

* programmer - steve hankin
* NOAA/PMEL, Seattle, WA - Tropical Modeling and Analysis Program
* written for VAX computer under VMS operating system
*
* V200: 5/26/89 - extracted from CONFINE_CONTEXT with improved error message
*	12/4/89 - improved error reporting
*	5/16/90 - further improved error reporting
*	 8/6/90 - error processing bug: "status" --> istat in SECS_FROM_BC
*       8/10/92 - don't eliminate transform because of lo=hi limits
* V230: 11/4/92 - expanded variable name length
* V420: 11/95 - change to philosophy of "explicit" contexts -- for
*		LET A=I[i=1:5:1] (dynamic axis) the expression A[I=10] should
*		be an error
*	      - allow explicit Z=100@ITP to pass through
*       - add check on cx_calendar rather than cx_lo/hi_ww to indicate whether time
*         is date or timesteps
* V491: 4/98 - fix bug in check for itsa_uvar -- was bypassing ALL processing
* V500 *kob* 3/99- up VAR_CODE  to 64 chars  and err_string from 48 to 64
*                  and VAR_TITLE from 24 to 128
* V520 *kob* 6/00 - had to had risc_buff text buffer for sgi output
* V530 *acm* 1/01 - add cal_id for alternative calendar types
* V552 *acm* 4/03- up VAR_CODE and err_str to 128 chars 
* V553 *acm* 8/03 - write an error message if time origin or time limits 
*                   were mis-defined.
* V554 *acm* 11/03 -Fix bug in the change just above: if times requested
*                   are outside the data bounds, give error message.
* V561 *acm*  4/04 -Fix bug plotting a noleap time axis: cx_lo_ww and cx_hi_ww
*                   are originally set without knowledge of calendar, so when
*                   resetting time step here, also reset the limits for cx_cmnd
* V580 *acm* 11/04  - Use cx_cal_id to keep track of which calendar 
*                     used when cx_lo_ww and cx_hi_ww are computed.
* V600 *acm* 11/05  - Fix for bug 1365; do not fix the world limits for cx_cmnd
*                     unless they were already set.
* V600 *acm*  4/06  - correct typo unspecified_val8
*      *acm*  4/12 6D Ferret: time axis may be in t or f direction.
* V680 *acm*  612  6D Ferret: Pull the calendar correction into a separate routine.
* V683  *acm*10/12 Improve handling of F axes and info about the calendar of the grid.
* V695  *acm* 2/15 cx_calendar stores info about calendar formatting of T and F axes
* V720  *sh*  4/17 support for DSG ragged contiguous grids
* V741 6/18 *acm* Handle modulo longitudes in DSG files
* V751 8/19 *acm* Issue 1954: For DSG grids allow for modulo longitudes for station locations
*                 of timeseries or profiles as well as trajectory data.
* V76  1/20 *acm* working with Point-type dsg data 

        IMPLICIT NONE
	include 'tmap_dims.parm'
	include 'implicit.parm'
	include	'xtm_grid.cmn_text'
	external xgt_grid_data
	include	'ferret.parm'
	include	'errmsg.parm'
	include	'xvariables.cmn'
	include	'xcontext.cmn'
	include	'xtext_info.cmn'
	include 'xrisc.cmn'

* calling argument declarations:
	INTEGER	cx, idim, status

* internal variable declarations:
!	LOGICAL lim_redundant
	LOGICAL TM_ITSA_DSG, ITSA_DSG_POINTS, its_irrelevant,
     .		its_dsg, its_cmpnd
	INTEGER	ISUBSCRIPT, TM_LENSTR1, TM_DSG_DSET_FROM_GRID,
     .		dset, ss, min_ss, max_ss, lo_ss, hi_ss,
     .		axis, grid, lenstr, slen, cat, var, istat,
     .		orientation, obs_dimlen, fline
	REAL*8	ww, min_ww, max_ww, lo_ww, hi_ww
*                  and VAR_TITLE from 24 to 128
	CHARACTER VAR_CODE*128, CX_DIM_STR*48, VAR_TITLE*128,
     .		  buff*48, err_str*128, expr_num*1

* internal parameter declarations:
	LOGICAL     full_precision
	INTEGER	    err_ss_lims, err_ww_lims
	PARAMETER ( full_precision = .TRUE.,
     .		    err_ss_lims = -77,
     .		    err_ww_lims = -88 )

* initialize
	grid = cx_grid( cx )
	cat  = cx_category(cx)
	var  = cx_variable(cx)
	its_dsg = TM_ITSA_DSG(grid) .OR. ITSA_DSG_POINTS(cx)
c	IF (its_dsg) THEN
c	   dset = TM_DSG_DSET_FROM_GRID( grid )
c	   CALL TM_DSG_FACTS( grid, orientation, obs_dimlen, fline, its_dsg, its_cmpnd )
c	ENDIF

* (11/95) expressions such as "2+2"
	IF (cat .EQ. cat_user_var) THEN
	   its_irrelevant = uvar_given(idim,var)
     .			.EQ. uvlim_irrelevant
	ELSE
	   its_irrelevant = .FALSE.
	ENDIF

* axis for this dimension
	axis = grid_line( idim, grid )

* check for internal error left over from old TMAP library
	IF ( axis .EQ. munknown ) THEN
	   status = ferr_grid_definition
	   GOTO 5010
	
! this block of code appears in FLESH_OUT_AXIS instead, because cannot
! constrain the T axis limits until after they are converted to taxis encoding
!* *** DSG grid (4/17)
!	ELSEIF ( its_dsg .AND. idim.NE.f_dim		) THEN
!	   dset = TM_DSG_DSET_FROM_GRID(grid)
!	   CALL DSG_COORD_LIMS(dset, idim, min_ww, max_ww)
!	   IF (min_ww .EQ. unspecified_val8) GOTO 1000   ! irrelevant axis
!* ... constrain to the ww datset extremes
!	   IF  (cx_lo_ww(idim,cx) .EQ. unspecified_val8) THEN
!	      cx_lo_ww(idim,cx) = min_ww
!	      cx_hi_ww(idim,cx) = max_ww
!	   ELSE
!	      IF (cx_lo_ww(idim,cx) .LT. min_ww) cx_lo_ww(idim,cx) = min_ww 
!	      IF (cx_hi_ww(idim,cx) .GT. max_ww) cx_hi_ww(idim,cx) = max_ww 
!	   ENDIF
!	   GOTO 1000

* irrelevant normal axis  (was based on lim_redundant prior to 11/95)
	ELSEIF ( axis .EQ. mnormal ) THEN
	   GOTO 1000

* (11/95) limits on a constant are irrelevant
	ELSEIF ( cat .EQ. cat_constant ) THEN
	   GOTO 1000

* (11/95) expressions such as "2+2"
	ELSEIF ( its_irrelevant ) THEN
	   GOTO 1000

* positions were specified by subscript
	ELSEIF ( cx_by_ss( idim, cx ) ) THEN
	   status = err_ss_lims	! in case of error
* ... is there valid position data for this axis ?
	   IF ( cx_lo_ss( cx ,idim ) .EQ. unspecified_int4 ) GOTO 1000
* ... confine them to the limits of the defining grid
	   CALL GRID_SUBSCRIPT_EXTREMES(min_ss, max_ss, grid, idim)
	   lo_ss = MAX( min_ss, cx_lo_ss(cx,idim) )
	   hi_ss = MIN( max_ss, cx_hi_ss(cx,idim) )
* ... check for limit error
	   IF ( lo_ss .GT. hi_ss ) GOTO 5010
* ... success
	   cx_lo_ss( cx, idim ) = lo_ss
	   cx_hi_ss( cx, idim ) = hi_ss

* positions were specified by world coordinates
	ELSE
	   status = err_ww_lims	! in case of error
* ... is there valid position data for this axis ?
	   IF ( cx_lo_ww( idim, cx ) .EQ. unspecified_val8 ) GOTO 1000
* ... time/date strings are received as -1*(secs from BC)
* 8/97 *kob* use cx_calender to check if time is date rather than time step
* 2/01 *acm* reset time step if axis is a different calendar than Gregorian.
* 8/03 *acm* write error message if time origin or limits were mis-defined.
*11/04 *acm* reset time step only if axis is a different calendar than 
*            the calendar originally used to compute world coordinates

	   IF ( (idim .EQ. t_dim .OR. idim .EQ. f_dim) .AND.cx_calendar(idim,cx) ) THEN

	      CALL CALENDAR_AXIS (idim, grid, cx, istat)
	      IF (istat .NE. 2) THEN
	         IF (istat .EQ. -1) GOTO 5000

	      ENDIF

	      cx_calendar(idim,cx) = .FALSE. 

           ENDIF
* ... confine values to the limits of the defining grid
 4000	   CALL GRID_WORLD_EXTREMES( min_ww, max_ww, grid, idim )
	   lo_ww = MAX( cx_lo_ww( idim, cx ), min_ww )
	   hi_ww = MIN( cx_hi_ww( idim, cx ), max_ww )

	   IF (its_dsg .AND. idim.EQ.x_dim) THEN  ! Moduloing done elsewhere
	      lo_ww = cx_lo_ww( idim, cx_cmnd ) 
	      hi_ww = cx_hi_ww( idim, cx_cmnd )
	   ENDIF

* ... check for error in limits and possibly remove transformation
	   IF ( lo_ww .GT. hi_ww ) THEN
	      GOTO 5010
	   ENDIF
* ... success
	   cx_lo_ww( idim, cx ) = lo_ww
	   cx_hi_ww( idim, cx ) = hi_ww

	ENDIF

* successful completion
 1000	status = ferr_ok
	RETURN

* error exits
 5000	RETURN
* prepare detailed error string: variable name and range along axis
 5010	err_str = VAR_CODE(cat,var)
	IF ( INDEX( err_str,'#' ) .GT. 0 ) THEN
	   err_str = '"'//VAR_TITLE( cx )
	   lenstr  = TM_LENSTR1(err_str)
	   IF ( lenstr .GT. 20 ) THEN
	      err_str = err_str(:20)//'..."'
	   ELSE
	      err_str = err_str(:lenstr)//'"'
	   ENDIF
	ENDIF
	lenstr = TM_LENSTR1(err_str)
	IF ( cx_unstand_grid(cx) ) THEN
	   err_str = err_str(:lenstr)//' on grid '//grid_name(cx_grid(cx))
	   lenstr = TM_LENSTR1(err_str)
	ENDIF
	CALL TRANSFER_CONTEXT( cx, cx_buff )
	cx_trans( idim, cx_buff ) = trans_no_transform
	IF ( status .EQ. err_ss_lims ) THEN
	   cx_lo_ss(cx_buff,idim) = min_ss
	   cx_hi_ss(cx_buff,idim) = max_ss
	   GOTO 5900
	ELSEIF ( status .EQ. err_ww_lims ) THEN
	   cx_lo_ww(idim,cx_buff) = min_ww
	   cx_hi_ww(idim,cx_buff) = max_ww
	   GOTO 5900
	ELSEIF ( status .EQ. ferr_grid_definition ) THEN
	   CALL ERRMSG( ferr_grid_definition, status,
     .			err_str(:lenstr)//pCR//
     .			'Unknown '//ww_dim_name(idim)//' axis in grid '//
     .			grid_name(grid), *5000 )
	ELSE
!	   STOP 'CONFINE_AXIS'
           GO TO 5000
	ENDIF

 5900	buff = CX_DIM_STR(idim, cx, ':',full_precision,slen)
*       Appending space just to get rid of a nonsense compiler warning
	risc_buff = buff(:slen)//' '
        IF ( cx_hi_ss(cx,idim) .NE. cx_lo_ss(cx,idim) ) THEN
	   CALL ERRMSG( ferr_limits, status,
     .		err_str(:lenstr)//' is not in the range '//risc_buff//
     .		pCR//'Axis extremes are '//CX_DIM_STR(idim, cx_buff, 
     .					  ':',full_precision,slen), *5000 )

        ELSE
	   CALL ERRMSG( ferr_limits, status,
     .		err_str(:lenstr)//' does not exist at '//risc_buff//
     .		pCR//'Axis extremes are '//CX_DIM_STR(idim, cx_buff, 
     .					  ':',full_precision,slen), *5000 )
        ENDIF

	END
