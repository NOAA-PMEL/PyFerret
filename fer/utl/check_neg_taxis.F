	SUBROUTINE CHECK_NEG_TAXIS( dset )

* check to make sure that the time axis of the data set doesn't have any
* negative time step values

* programmer - steve hankin
* NOAA/PMEL, Seattle, WA - Tropical Modeling and Analysis Program
* written for VAX computer under VMS operating system
*
* V200:  6/11/90
* Unix/RISC port:  07/91 - allow normal time axis
*                  10/91 - briefer warning
* V230 - simplified (less intimidating) message
*      7/20/92 - don't issue warning unless T0 is valid
*              - (cleaned up unused variables)

* calling argument declarations:
	INTEGER	dset

* internal variable declarations:
        LOGICAL TM_DATE_OK
	INTEGER	i, grid, line
	DOUBLE PRECISION TM_WORLD

#ifdef unix
	include 'ferret_cmn/ferret.parm'
	include 'tmap_format/tmap_dims.parm'
	include 'tmap_format/xdset_info.cmn_text'
	external xdset_info_data
	include 'tmap_format/xtm_grid.cmn_text'
	external xgt_grid_data
#else
	INCLUDE 'FERRET_CMN:FERRET.PARM'
	INCLUDE 'TMAP_FORMAT:TMAP_DIMS.PARM'
	INCLUDE 'TMAP_FORMAT:XDSET_INFO.CMN'
	INCLUDE 'TMAP_FORMAT:XTM_GRID.CMN'
#endif

* base it on first variable in data set
	DO 100 i = 1, maxvars
 100	IF ( ds_var_setnum(i) .EQ. dset ) GOTO 110
	RETURN	! no vars in set

* check lowest time step value on axis
 110	grid = ds_grid_number(i)
        line = grid_line(t_dim, grid )

* allow time-independent grids (7/91)
        IF ( line .EQ. mnormal ) RETURN

* no warning for non-date axes - this way time can be a regular ol' axis
        IF ( .NOT.TM_DATE_OK( line_t0(line) ) ) RETURN

	IF ( TM_WORLD(1,grid,t_dim,box_middle) .LT. 0.0D0 ) THEN
	   CALL WARN('Negative values in time axis'//
     .               ' may not be formatted properly: '//
     .		     ds_name(dset) )

	ELSEIF ( TM_WORLD(1,grid,t_dim,box_lo_lim) .LT. 0.0D0 ) THEN
	   CALL WARN('Negative values at start of time axis'//
     .               ' may be improperly formatted: '//
     .		     ds_name(dset) )
	ENDIF

	RETURN
	END
