	CHARACTER*14 FUNCTION ABS_WHOI_DATE( secsfrombc )

* create a string of the form YYMMDDHHMM from the absolute time in seconds

* programmer - steve hankin
* NOAA/PMEL, Seattle, WA - Tropical Modeling and Analysis Program

* V300: 10/93 *sh*  most of the code lifted from WHOI_DATE
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
	REAL*8	secsfrombc

* internal variable declarations
	INTEGER 	yr100, yr, mon, day, hr, min, sec,
     .			status, i, axis
	CHARACTER*20	TM_SECS_TO_DATE, date

* Convert to date string
	date = TM_SECS_TO_DATE(secsfrombc)

* break the date string into component integer values
	CALL TM_BREAK_DATE( date, yr, mon, day, hr, min, sec, status )
	yr100 = yr/100		! 1980 --> 19
	yr = yr - (yr100*100)	! 1982 --> 82

* assemble the pieces in WHOI format
	WRITE (ABS_WHOI_DATE,3000) yr100,yr,mon,day,hr,min,sec
 3000	FORMAT (7I2.2)

	RETURN
	END
