	LOGICAL FUNCTION VALID_GRID( grid )

* check to see that all the axes of the given grid are well defined

* programmer - steve hankin
* NOAA/PMEL, Seattle, WA - Tropical Modeling and Analysis Program
* written for VAX computer under VMS operating system
*
* V200: 10/24/89

* calling argument declarations:
	INTEGER	grid

* internal variable declarations:
	INTEGER	idim

#ifdef unix
	include 'tmap_format/tmap_dims.parm'
	include	'tmap_format/xtm_grid.cmn_text'
	external xgt_grid_data
	include	'ferret_cmn/ferret.parm'
#else
	INCLUDE 'TMAP_FORMAT:TMAP_DIMS.PARM'
	INCLUDE	'TMAP_FORMAT:XTM_GRID.CMN'
	INCLUDE	'FERRET_CMN:FERRET.PARM'
#endif

	DO 100 idim = 1, 4
	   IF ( grid_line(idim,grid) .EQ. munknown ) THEN
* ... its bad
	      VALID_GRID = .FALSE.
	      RETURN
	   ENDIF
 100	CONTINUE

* its OK
	VALID_GRID = .TRUE.
	RETURN
	END
