	DOUBLE PRECISION FUNCTION SECS_TO_TSTEP( grid, secs )

* convert an absolute time measured in seconds from 1-JAN-0000:00:00:00
* into a time step value relevant on a particular grid, "grid"

* programmer - steve hankin
* NOAA/PMEL, Seattle, WA - Tropical Modeling and Analysis Program
* written for VAX computer under VMS operating system
*
* revision 0.0 - 7/27/88

#ifdef unix
	include 'tmap_format/tmap_dims.parm'
        include 'tmap_format/xtm_grid.cmn_text'
	external xgt_grid_data
        include 'ferret_cmn/ferret.parm'
#else
	INCLUDE 'TMAP_FORMAT:TMAP_DIMS.PARM'
        INCLUDE 'TMAP_FORMAT:XTM_GRID.CMN'
        INCLUDE 'FERRET_CMN:FERRET.PARM'
#endif

* calling argument declarations:
	INTEGER	grid
	REAL*8	secs

* internal variable declarations:
	INTEGER		axis, status
	CHARACTER*20	date
	REAL*8		SECS_FROM_BC, bc_to_t0

* determine time axis of grid
	axis = grid_line( t_dim, grid )

* valid time axis ?
	IF ( line_direction( axis ) .EQ. 'TI' ) THEN

* seconds from BC to start of time axis
	   bc_to_t0 = SECS_FROM_BC( line_t0(axis), status )

* convert to units used on this time axis
	   SECS_TO_TSTEP = ( secs - bc_to_t0 ) / line_tunit(axis)

	ELSE

* not a valid time axis
	   SECS_TO_TSTEP = secs		! no conversion possible

	ENDIF
	RETURN
	END
