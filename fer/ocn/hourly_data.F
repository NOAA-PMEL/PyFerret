	LOGICAL FUNCTION HOURLY_DATA( grid )

* determine whether a particular grid has a regular hourly time axis

* programmer - steve hankin
* NOAA/PMEL, Seattle, WA - Tropical Modeling and Analysis Program
* written for VAX computer under VMS operating system

* V200:  7/27/89

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

* internal variable declarations:
	INTEGER	taxis

	taxis = grid_line( t_dim, grid )

	IF ( taxis .EQ. munknown .OR. taxis.EQ.mnormal ) THEN
	   HOURLY_DATA = .FALSE.
	   RETURN
	ENDIF

	HOURLY_DATA =
     .		 line_regular  ( taxis )
     .	   .AND. line_delta    ( taxis ) .EQ. 1.
     .	   .AND. line_tunit    ( taxis ) .EQ. 3600.
     .	   .AND. line_dim      ( taxis ) .GT. 1
     .	   .AND. line_direction( taxis ) .EQ. 'TI'

	RETURN
	END
