	CHARACTER*14 FUNCTION WHOI_DATE( grid, tstep )

* create a string of the form YYYYMMDDHHMMSS (was YYMMDDHHMM) from the time step

* programmer - steve hankin
* NOAA/PMEL, Seattle, WA - Tropical Modeling and Analysis Program

* revision 0.0 - 3/15/88
* revision 0.1 - 1/18/89 - BREAK_DATE --> TM_BREAK_DATE
* V200:  1/31/90 - fixed bug with climatological axes (years "0000" and "0001"
*		   but produce the same WHOI dates)
* V420: 11/27/95 - 4-digit YEARS!!

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

* calling argument declarations
	INTEGER	grid
	REAL*8	tstep

* internal variable declarations
	INTEGER 	yr100, yr, mon, day, hr, min, sec,
     .			status, i, axis
	CHARACTER*20	TM_SECS_TO_DATE, date
	REAL*8		start_secs, offset_secs, this_secs
	REAL*8		SECS_FROM_BC

* get the axis number
	axis = grid_line( t_dim, grid )
	IF ( axis .EQ. mnormal .OR. axis .EQ. munknown ) THEN
	   WHOI_DATE = '00000000000000'
	   RETURN
	ELSEIF ( line_direction(axis) .NE. 'TI' ) THEN
	   WHOI_DATE = '00000000000000'
	   RETURN
	ENDIF

* First find the offset from 01-JAN-0000 00:00:00 from the descriptor file
	start_secs = SECS_FROM_BC( line_t0(axis), status )

* Find the number of seconds since start of data set for this time step
	offset_secs = tstep*line_tunit( axis )

* Total seconds to this time step since 01-JAN-0000 00:00:00
	this_secs = start_secs + offset_secs

* Convert to date
	date = TM_SECS_TO_DATE(this_secs)

* break the date string into component integer values
	CALL TM_BREAK_DATE( date, yr, mon, day, hr, min, sec, status )
	yr100 = yr/100		! 1980 --> 19
	yr = yr - (yr100*100)	! 1982 --> 82

* assemble the pieces in WHOI format
	WRITE (WHOI_DATE,3000) yr100,yr,mon,day,hr,min,sec
 3000	FORMAT (7I2.2)

	RETURN
	END
