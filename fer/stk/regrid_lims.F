	SUBROUTINE REGRID_LIMS( dst_cx, src_cx, axis, status, * )

* given a context describing data to be regridded to a destination grid
* - select one of the axes needing regridding
* - create, if necessary, an intermediate grid that differs from the
*	destination grid by only this single axis
* - compute the region on the source grid (possibly the newly created
*	intermediate grid) that will supply the re-griddable data
* When called the source context should be a copy of the destination context 

* programmer - steve hankin
* NOAA/PMEL, Seattle, WA - Tropical Modeling and Analysis Program
* written for VAX computer under VMS operating system
*
* revision 0.0 -  7/25/88
* revision 0.1 -   9/1/89 - corrected bug in destination world limits
*			  - added xy averaging
* revision 0.2 -   6/2/89 - modified grid limits calc. for modulo axes
* V200:  7/31/89 - 4D symmetrical version
*		 - fixed bug following GRID_SUBSCRIPT_EXTREMES for modulo axes
*	 11/3/89 - if destination limits are unknown flag error
*	 12/1/89 - handles XY averaging - (uses RQST_REGRID)
* V420: 10/95	 - Support for implicit axes: intermed. grids inherit
*		   responsibilities (use counts) for implicit axes
*	11/95	   check that destination axis is modulo for modulo regrid
*		   and take abnormal return if it dest region is
*		   modulo-wrapped
* V440: 9/96	 - Bug: failed to call TM_USE_LINE for dynamic line in
*		   intermediate grid (err430_deallo_dyn_line)
*		   Also corrected undetected bug: in intermediate grid for
*		   XY averaging need to deallo the y_dim not the "axis"

#ifdef unix
	include 'tmap_format/tmap_dims.parm'
	include	'tmap_format/xtm_grid.cmn_text'
	external xgt_grid_data
	include	'ferret_cmn/ferret.parm'
	include	'ferret_cmn/errmsg.parm'
	include	'ferret_cmn/xcontext.cmn'
	include	'ferret_cmn/xvariables.cmn'
	include	'ferret_cmn/xtext_info.cmn'
#else
	INCLUDE 'TMAP_FORMAT:TMAP_DIMS.PARM'
	INCLUDE	'TMAP_FORMAT:XTM_GRID.CMN'
	INCLUDE	'FERRET_CMN:FERRET.PARM'
	INCLUDE	'FERRET_CMN:ERRMSG.PARM'
	INCLUDE	'FERRET_CMN:XCONTEXT.CMN'
	INCLUDE	'FERRET_CMN:XVARIABLES.CMN'
	INCLUDE	'FERRET_CMN:XTEXT_INFO.CMN'
#endif

* calling argument declarations:
	INTEGER	src_cx, dst_cx, status

* internal variable declarations:
	LOGICAL	ITSA_MODULO_REGRID, ITSA_MODULO_AXIS, last_axis, xy_ave
	INTEGER	KNOWN_GRID, CAXIS_LEN,
     .		idim, axis, var_grid, src_grid, dst_grid, next_axis
	CHARACTER*24 VAR_CODE

* insert source grid ( defining grid for this variable )
	dst_grid = cx_grid( dst_cx )
	var_grid = KNOWN_GRID( cx_data_set(src_cx),
     .			  cx_category(src_cx),
     .			  cx_variable(src_cx)	)

* select an axis for regridding
	DO 10 axis = 1, 4
 10	IF (grid_line(axis,dst_grid).NE.grid_line(axis,var_grid)) GOTO 20
* ... signal calling routine that there's nothing to regrid
	cx_unstand_grid( dst_cx ) = .FALSE.
	status = ferr_ok
	RETURN

* special requirement for modulo regrid
 20	IF ( ITSA_MODULO_REGRID(dst_cx,axis) ) THEN
	   IF ( .NOT.ITSA_MODULO_AXIS(dst_cx,axis) ) GOTO 5100
* ... strip off the destination modulo wraps if any with abnormal return
* (so we compute just the base limits, 1-N, of the modulo axis)
	   IF ( cx_lo_ss(dst_cx,axis) .NE. unspecified_int4 ) THEN
	      IF (cx_hi_ss(dst_cx,axis ) .GT. CAXIS_LEN(axis,dst_cx)
     .       .OR. cx_lo_ss(dst_cx,axis ) .LT. 1 ) RETURN 1
	   ENDIF
	ENDIF

* is this a 2D (XY averaging) regrid?
	xy_ave = cx_regrid_trans(axis,src_cx) .EQ. prgrd_xy_ave
	IF ( xy_ave ) THEN
	   next_axis = z_dim
	ELSE
	   next_axis = axis + 1
	ENDIF

* is this the last axis to regrid ?
	last_axis = .TRUE.
	DO 25 idim = next_axis, 4
 25	IF ( grid_line(idim,dst_grid) .NE. grid_line(idim,var_grid) )
     .							last_axis = .FALSE.

* if its the last axis then the source data will be on its defining grid
	IF ( last_axis ) THEN
	   src_grid = var_grid
	   cx_unstand_grid( src_cx ) = .FALSE.
	   is_phase( isp )   = prgrd_from_source
	ELSE		
* we need an intermediate grid
	   cx_unstand_grid( src_cx ) = .TRUE.
	   CALL ALLO_GRID( src_grid,status )
	   IF ( status .NE. ferr_ok ) GOTO 5000
	   CALL TM_COPY_GRID_W_LINE_USE( dst_grid, src_grid ) 
	   grid_name( src_grid ) = ww_dim_name(axis)//'NTERMED'	! for diag out
	   IF (grid_line(axis,src_grid) .GT. max_lines) CALL
     .				TM_DEALLO_DYN_LINE(grid_line(axis,src_grid))
	   grid_line( axis, src_grid ) = grid_line( axis, var_grid )
	   IF (grid_line(axis,var_grid) .GT. max_lines) CALL
     .				TM_USE_LINE(grid_line(axis,var_grid)) ! 9/96
	   IF (xy_ave) THEN
! 9/96: changed DEALLO grid_line(axis,src_grid) to grid_line(y_dim,src_grid)
	      IF (grid_line(y_dim,src_grid) .GT. max_lines) CALL
     .				TM_DEALLO_DYN_LINE(grid_line(y_dim,src_grid))
	      grid_line(y_dim,src_grid) = grid_line(y_dim,var_grid)
	      IF (grid_line(y_dim,var_grid) .GT. max_lines) CALL
     .				TM_USE_LINE(grid_line(y_dim,var_grid)) !9/96
	   ENDIF
	   is_phase( isp ) = prgrd_from_ntrmed
	ENDIF
	cx_grid( src_cx ) = src_grid

* get limits to request
	CALL RQST_REGRID( dst_cx, src_cx, axis, status )
	IF ( status .NE. ferr_ok ) GOTO 5000
	IF (xy_ave) CALL RQST_REGRID( dst_cx, src_cx, y_dim, status ) 
	IF ( status .NE. ferr_ok ) GOTO 5000

* success
	RETURN

* error exit
 5000	RETURN
 5100	CALL ERRMSG( ferr_regrid, status,
     .          '@MOD regridding to a non-modulo axis: '
     .          //VAR_CODE(cx_category(dst_cx),cx_variable(dst_cx)),
     .		*5000)

	END
