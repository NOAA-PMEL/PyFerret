	SUBROUTINE FLESH_OUT_AXIS( idim, cx, status )

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
* given a skeletal context with a known grid transfor the subscript coord
* data to world coordinates and world to subscript on the specified axis
* make sure all time coordinates are in time step values (vs. secs from BC)

* programmer - steve hankin
* NOAA/PMEL, Seattle, WA - Tropical Modeling and Analysis Program
* written for VAX computer under VMS operating system
*
* V200:  5/26/89 - extracted from FLESH_OUT_CONTEXT
*                - handles axes of uvars where all region info is contained
*		 - detects and set up interpolations
*		 - inserts unspecified limits for file vars
* V230:  7/20/92 - check TM_DATE_OK for valid time axis before doing time xlate
*                - kludge fix to err230_fixed_limits.jnl
*        8/10/92 - point location with transform (e.g. Z=5@DIN) needs
*                  exact location preserved for proper labelling
* V320:	 11/2/94 - checking if interpolation is required needs to be based on
*		   single precision coordinate comparison
* V420:   1/96   - if the context being completed involves modulo
*		   regridding then the interpretation of the limits is altered
* *kob* 11/96   - Linux port.  Linux didn't like tabs between "include"
*                 and what was being included.  removed them
* V450:   1/97   - fix to never-observed bug: the check
*		        "uvar_given(idim,var) .LE. uvlim_gvn_xact"
*		   ignores the possibility of uvlim_irrelevant
*  *KOB* 9/97    - replaced call to ISUBSCRIPT w/ call to ISUBSCR_CX in 
*                  order to use cx_calendar to determine style of taxis
* V500 *kob* 3/99- up VAR_CODE decl from 8 to 64 chars

#ifdef unix
	include 'tmap_dims.parm'
	include 'xtm_grid.cmn_text'
	external xgt_grid_data
	include 'ferret.parm'
	include 'errmsg.parm'
	include 'xcontext.cmn'
	include 'xvariables.cmn'
	include 'xprog_state.cmn'
	include 'xfr_grid.cmn'
	include 'xtext_info.cmn'
#else
	INCLUDE 'TMAP_FORMAT:TMAP_DIMS.PARM'
	INCLUDE	'TMAP_FORMAT:XTM_GRID.CMN'
	INCLUDE	'FERRET_CMN:FERRET.PARM'
	INCLUDE	'FERRET_CMN:ERRMSG.PARM'
	INCLUDE	'FERRET_CMN:XCONTEXT.CMN'
	INCLUDE	'FERRET_CMN:XVARIABLES.CMN'
	INCLUDE 'FERRET_CMN:XPROG_STATE.CMN'
	INCLUDE 'FERRET_CMN:XFR_GRID.CMN'
	INCLUDE 'FERRET_CMN:XTEXT_INFO.CMN'
#endif

* calling argument declarations:
	INTEGER	idim, cx, status

* internal variable declarations:
	LOGICAL TM_DATE_OK, TM_FPEQ, INHERITED_LIMITS, point, lim_redundant
	INTEGER	TM_LENSTR1, ISUBSCR_CX,
     .		ss, lo_ss, hi_ss, axis, grid, cat, var, slen
	REAL*8	TM_WORLD, SECS_TO_TSTEP, ww_nrst, ww_given
* kob 3/99 up VAR_CODE from 8 to 64 chars
	CHARACTER	VAR_TRANS*40, VAR_CODE*64, string*40

* initialize
	var  = cx_variable( cx )
	cat  = cx_category( cx )
	grid = cx_grid    ( cx )

* check for special actions needed on user-defined variables
* ( e.g. how to handle U[K=5] - U[K=1:10] )
	IF ( cat .EQ. cat_user_var ) THEN
	   lim_redundant = uvar_given( idim, var ) .LE. uvlim_gvn_xact
     .		     .OR.  uvar_given( idim, var ) .EQ. uvlim_irrelevant ! 1/97
	ELSE
	   lim_redundant = .FALSE.
	ENDIF

* for modulo regridding the apparent limits supplied in the definition
* are "swallowed" up by the regridding process. Externally applied limits
* **should** be applied (e.g. A[GT=climatology@MOD,L=1:3] where A=SST[L=lo:hi])
	IF ( .NOT.INHERITED_LIMITS(idim,cx) )  lim_redundant = .FALSE.

* axis for this dimension
	axis = grid_line( idim, grid )

	IF ( axis .EQ. mnormal ) THEN
* THIS AXIS IS NORMAL TO THE DEFINING GRID
* wipe clean any stray information about it
	   cx_lo_ss( cx, idim ) = unspecified_int4
	   cx_hi_ss( cx, idim ) = unspecified_int4
	   cx_lo_ww( idim, cx ) = unspecified_val8
	   cx_hi_ww( idim, cx ) = unspecified_val8
	   cx_delta( idim, cx ) = unspecified_val8
	   cx_trans( idim, cx ) = trans_no_transform
* kob 9/97 - need to reset cx_calendar as well
	   IF (idim .eq. t_dim) 
     .                cx_calendar(cx) = .FALSE.

* axes with fully "given" regions will have their regions determined via
* MERGE_CONTEXT (when computed) or PASS_AMBIG_LIMS (when found in memory)
	ELSEIF ( .NOT.cx_given(idim,cx) .AND. lim_redundant ) THEN
****
* 7/92 kludge:
* this routine is called by is_uvar_grid (while determining the grid)
* BEFORE uvar_given has been determined.  When called from there lim_redundant
* may be invalid.  Since uvar_given is initialized to uvlim_irrelevant we will
* use this as value a test of whether the call is from is_uvar_grid
* Probably it would be better to make this check while determining
* lim_redundant, above, but that would be a more significant alteration of
* working code - so this is the chicken-out solution
* WAS:	   IF ( cx_trans(idim,cx) .NE. trans_no_transform ) GOTO 5000

* 1/97 - might this kludge have been rendered unnecessary by the 1/97 change to
*        the determination of lim_redundant??
****
	   IF ( cx_trans(idim,cx)    .NE. trans_no_transform
     .    .AND. uvar_given(idim,var) .NE. uvlim_irrelevant ) GOTO 5000
****
	   cx_lo_ss( cx, idim ) = unspecified_int4
	   cx_hi_ss( cx, idim ) = unspecified_int4
	   cx_lo_ww( idim, cx ) = unspecified_val8
	   cx_hi_ww( idim, cx ) = unspecified_val8
	   cx_delta( idim, cx ) = unspecified_val8
	   cx_trans( idim, cx ) = trans_no_transform
* kob 9/97 - need to reset cx_calendar as well
	   IF (idim .eq. t_dim) 
     .                cx_calendar(cx) = .FALSE.

	ELSEIF ( cx_by_ss( idim, cx ) ) THEN
* POSITIONS WERE SPECIFIED BY SUBSCRIPT
* is there valid position data for this axis ?
	   IF ( cx_lo_ss( cx, idim ) .EQ. unspecified_int4 ) THEN
* ... no position data but maybe it's a file variable with only 1 point
	      CALL VAR_SS_LIMS( idim, cx, lo_ss, hi_ss )
! old	      IF ( lo_ss.EQ.unspecified_int4 .OR. lo_ss.NE.hi_ss ) RETURN
	      IF ( lo_ss.EQ.unspecified_int4 ) RETURN
	      cx_lo_ss(cx,idim) = lo_ss
	      cx_hi_ss(cx,idim) = hi_ss
	    ENDIF

* compute world positions as grid box edges
	   cx_lo_ww(idim, cx) = TM_WORLD( cx_lo_ss( cx,idim),
     .					  grid, idim, box_lo_lim )
	   cx_hi_ww(idim, cx) = TM_WORLD( cx_hi_ss( cx,idim),
     .					  grid, idim, box_hi_lim )

* POSITIONS WERE SPECIFIED BY WORLD COORDINATES
	ELSE
* is there valid position data for this axis ?
	   IF ( cx_lo_ww( idim, cx ) .EQ. unspecified_val8 ) THEN
* ... no position data but maybe it's a file variable with only 1 point
	      CALL VAR_SS_LIMS( idim, cx, lo_ss, hi_ss )
! old	      IF ( lo_ss.EQ.unspecified_int4 .OR. lo_ss.NE.hi_ss ) RETURN
	      IF ( lo_ss.EQ.unspecified_int4 ) RETURN
	      cx_lo_ss(cx,idim) = lo_ss
	      cx_hi_ss(cx,idim) = hi_ss
* ... compute world positions as grid box edges
	      cx_lo_ww(idim, cx) = TM_WORLD( lo_ss, grid, idim, box_lo_lim )
	      cx_hi_ww(idim, cx) = TM_WORLD( hi_ss, grid, idim, box_hi_lim )
	    ENDIF

	   point = cx_lo_ww(idim,cx) .EQ. cx_hi_ww(idim,cx)

* convert times given as seconds from BC to time step values
* *kob* 8/97 - use cx_calender instead of cx_lo_ww
	   IF ( idim.EQ.t_dim .AND. cx_calendar(cx) ) THEN
              IF ( TM_DATE_OK( line_t0(axis) ) ) THEN
	         cx_lo_ww(idim,cx) =
     .                SECS_TO_TSTEP( grid, -cx_lo_ww(idim,cx) )
	         cx_hi_ww(idim,cx) =
     .                SECS_TO_TSTEP( grid, -cx_hi_ww(idim,cx) )
		 cx_calendar(cx) = .FALSE.
              ENDIF
	   ENDIF
	      
* world coordinate is a single point
	   IF ( point ) THEN
	      ww_given = cx_lo_ww(idim,cx)
	      ss = ISUBSCR_CX(ww_given,grid,idim,cx,round_dn)
	      ww_nrst = TM_WORLD( ss, grid, idim, box_middle )
	      CALL GRID_SUBSCRIPT_EXTREMES( lo_ss, hi_ss, grid, idim )
! ???	      CALL VAR_SS_LIMS( idim, cx, lo_ss, hi_ss )
* ... need to interpolate ? 
* Note: changed to TM_FPEQ(SNGL(xxx),...) 1/94  - testing indicates that
*		double precision time ranges are nexer used at this point
	      IF (  mode_interpolate
     .	     .AND. .NOT.TM_FPEQ(SNGL(ww_given),SNGL(ww_nrst)) ) THEN
* ... set up context to interpolate
	         IF ( cx_trans(idim,cx) .NE. trans_no_transform
     .		.AND. cx_trans(idim,cx) .NE. trans_interpolate ) GOTO 5200
	         cx_trans(idim,cx) = trans_interpolate
	      ELSEIF ( cx_trans(idim,cx) .EQ. trans_no_transform ) THEN
* ... interpolation not desired, needed or possible
	         cx_lo_ww(idim,cx) = ww_nrst
	         cx_hi_ww(idim,cx) = ww_nrst
	         cx_lo_ss(cx,idim) = ss
	         cx_hi_ss(cx,idim) = ss
	      ELSE 
* ... keep original (exact) limits if transformed axis ! 8/92
	         cx_lo_ss(cx,idim) = ss
	         cx_hi_ss(cx,idim) = ss
	      ENDIF
	   ELSE

* world coordinates span a range
* ... compute subscripts ( rounding inward to resolve ambiguities )
	      cx_lo_ss( cx, idim ) = ISUBSCR_CX
     .				(cx_lo_ww(idim, cx), grid, idim, cx,round_up)
	      cx_hi_ss( cx, idim ) = ISUBSCR_CX 
     .				(cx_hi_ww(idim, cx), grid, idim, cx,round_dn)
	   ENDIF
	ENDIF

* TRANSFORMATION REDUCES RANGE TO A POINT ?
	IF ( cx_trans(idim,cx) .GT. trans_compress_code ) THEN
	   cx_lo_ss( cx, idim ) = unspecified_int4
	   cx_hi_ss( cx, idim ) = unspecified_int4
	   cx_by_ss( idim, cx ) = .FALSE.
	ENDIF
	   
* successful completion
	status = ferr_ok
	RETURN

* error exits
 5000	RETURN
! 5100	string = VAR_TRANS( idim, cx, slen )
!	CALL ERRMSG( ferr_trans_nest, status,
!     .			'DEFINEd var has fixed limits '//string(:slen), *5000 )
 5200	string = VAR_CODE( cx_category(cx), cx_variable(cx) )
	CALL ERRMSG( ferr_trans_nest, status,
     .		'cannot perform interpolation '//pCR//
     .		ww_dim_name(idim)//' axis of '//string(:TM_LENSTR1(string))//
     .		' is already transformed'//pCR//
     .		'Can you define an intermediate variable ?',
     .		*5000 )
	END
