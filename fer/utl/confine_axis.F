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
* $Id$ 
*       - add check on cx_calendar rather than cx_lo/hi_ww to indicate whether time
*         is date or timesteps
* V491: 4/98 - fix bug in check for itsa_uvar -- was bypassing ALL processing
* V500 *kob* 3/99- up VAR_CODE  to 64 chars  and err_string from 48 to 64
*                  and VAR_TITLE from 24 to 128
* V520 *kob* 6/00 - had to had risc_buff text buffer for sgi output
#ifdef unix
	include 'tmap_dims.parm'
	include	'xtm_grid.cmn_text'
	external xgt_grid_data
	include	'ferret.parm'
	include	'errmsg.parm'
	include	'xvariables.cmn'
	include	'xcontext.cmn'
	include	'xtext_info.cmn'
	include 'xrisc.cmn'
#else
	INCLUDE 'TMAP_FORMAT:TMAP_DIMS.PARM'
	INCLUDE	'TMAP_FORMAT:XTM_GRID.CMN'
	INCLUDE	'FERRET_CMN:FERRET.PARM'
	INCLUDE	'FERRET_CMN:ERRMSG.PARM'
	INCLUDE	'FERRET_CMN:XVARIABLES.CMN'
	INCLUDE	'FERRET_CMN:XCONTEXT.CMN'
	INCLUDE	'FERRET_CMN:XTEXT_INFO.CMN'
	INCLUDE	'FERRET_CMN:XRISC.CMN'
#endif

* calling argument declarations:
	INTEGER	cx, idim, status

* internal variable declarations:
!	LOGICAL lim_redundant
	LOGICAL its_irrelevant
	INTEGER	ISUBSCRIPT, TM_LENSTR1, ss, min_ss, max_ss, lo_ss, hi_ss,
     .		axis, grid, lenstr, slen, cat, var, istat
	REAL*8	SECS_FROM_BC, bc_to_t0, ww, min_ww, max_ww, lo_ww, hi_ww
* V500 *kob* 3/99- up VAR_CODE  to 64 chars  and err_string from 48 to 64
*                  and VAR_TITLE from 24 to 128
	CHARACTER VAR_CODE*64, CX_DIM_STR*48, VAR_TITLE*128,
     .		  buff*48, err_str*64, expr_num*1

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
* (11/95) expressions such as "2+2"
	IF (cat .EQ. cat_user_var) THEN
	   its_irrelevant = uvar_given(idim,var)
     .			.EQ. uvlim_irrelevant
	ELSE
	   its_irrelevant = .FALSE.
	ENDIF

* axis for this dimension
	axis = grid_line( idim, grid )

! 11/95: increased restrictions that limits must be within valid bounds
!        even redundant limits must be fleshed out since they may be used at
!	 higher levels to "reconcile" results
!* check for special actions needed on user-defined variables
!* ( e.g. how to handle U[K=5] - U[K=1:10] or PI=3.1415)
!	IF ( cat .EQ. cat_user_var ) THEN
!	   lim_redundant = uvar_given( idim, var ) .LE. uvlim_gvn_xact
!	ELSE
!	   lim_redundant = .FALSE.
!	ENDIF

* check for internal error left over from old TMAP library
	IF ( axis .EQ. munknown ) THEN
	   status = ferr_grid_definition
	   GOTO 5010
	
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
	   IF ( idim .EQ. t_dim 
     .	  .AND.cx_calendar(cx) ) THEN
	      bc_to_t0 = SECS_FROM_BC( line_t0(axis), istat )
	      cx_lo_ww(idim,cx) = 
     .			-(cx_lo_ww(idim,cx)+bc_to_t0)/line_tunit(axis)
	      cx_hi_ww(idim,cx) = 
     .			-(cx_hi_ww(idim,cx)+bc_to_t0)/line_tunit(axis)
	      cx_calendar(cx) = .FALSE. 
	   ENDIF
* ... confine values to the limits of the defining grid
	   CALL GRID_WORLD_EXTREMES( min_ww, max_ww, grid, idim )
	   lo_ww = MAX( cx_lo_ww( idim, cx ), min_ww )
	   hi_ww = MIN( cx_hi_ww( idim, cx ), max_ww )
* ... check for error in limits and possibly remove transformation
	   IF ( lo_ww .GT. hi_ww ) THEN
	      GOTO 5010
! commented out 11/95 -- allow user's "@ITP" to pass through
!	   ELSEIF ( lo_ww .EQ. hi_ww
!     .	      .AND. cx_trans(idim,cx) .EQ. trans_interpolate ) THEN
!! ... B4 8/92 was: .AND. cx_trans(idim,cx) .GT. trans_compress_code ) THEN
!* ... remove transformation if irrevelent
!	      cx_trans( idim, cx ) = trans_no_transform
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
	   STOP 'CONFINE_AXIS'
	ENDIF

 5900	buff = CX_DIM_STR(idim, cx, ':',full_precision,slen)
	risc_buff = buff(:slen)
	CALL ERRMSG( ferr_limits, status,
     .		err_str(:lenstr)//' is not in the range '//risc_buff//
     .		pCR//'Axis extremes are '//CX_DIM_STR(idim, cx_buff, 
     .					  ':',full_precision,slen), *5000 )
	END
