	INTEGER FUNCTION DATE_PREC( grid, dt )

* determine the precision level needed for a date string to resolve "dt"
* in units appropriate for the given grid time axis

* programmer - steve hankin
* NOAA/PMEL, Seattle, WA - Tropical Modeling and Analysis Program
* written for VAX computer under VMS operating system
*
* revision 0.0 - 3/18/88

#ifdef unix
	include 'tmap_format/tmap_dims.parm'
	include 'tmap_format/xtm_grid.cmn_text'
	external xgt_grid_data
	include	'ferret_cmn/ferret.parm'
#else
	INCLUDE 'TMAP_FORMAT:TMAP_DIMS.PARM'
	INCLUDE 'TMAP_FORMAT:XTM_GRID.CMN'
	INCLUDE	'FERRET_CMN:FERRET.PARM'
#endif

* calling argument declarations
	INTEGER	grid
	REAL	dt

* internal variable declarations:
	INTEGER	i
	REAL	convert(5), delta
	DATA	convert(1)/ 60. /,	! sec --> min
     .		convert(2)/ 60. /,	! min --> hour
     .		convert(3)/ 24. /,	! hr  --> day
     .		convert(4)/ 30. /,	! day --> mon
     .		convert(5)/ 12. /	! mon --> year

* get the dt interval expressed in seconds
	delta = dt * line_tunit( grid_line(t_dim,grid) )

* see how big it is
	DO 100 i = 1, 5
	   delta = delta / convert(i)
	   IF ( delta .LT. 1.0 ) THEN
	      DATE_PREC = 7-i
	      RETURN
	   ENDIF
 100	CONTINUE

* very big ! use years, only to label
	DATE_PREC = 1
	RETURN
	END
