	REAL FUNCTION AXIS_DISTANCE ( lo_val, hi_val, idim, grid )

* compute the length of axis between the low and high limits given 
* converted to uniform units ( m ) in x, y and z

* programmer - steve hankin
* NOAA/PMEL, Seattle, WA - Tropical Modeling and Analysis Program
* written for VAX computer under VMS operating system
*
* revision 0.0 - 1/12/87
* revision 0.1 - 6/3/87  - changed km --> m as standard unit
* revision 0.2 - 4/18/88 - passed grid to get unit conversion factor

#ifdef unix
	include 'tmap_format/tmap_dims.parm'
	include	'tmap_format/xtm_grid.cmn_text'
	external xgt_grid_data
	include	'tmap_format/xunits.cmn_text'
	external xunits_data
	include	'ferret_cmn/ferret.parm'
#else
	INCLUDE 'TMAP_FORMAT:TMAP_DIMS.PARM'
	INCLUDE	'TMAP_FORMAT:XTM_GRID.CMN'
	INCLUDE	'TMAP_FORMAT:XUNITS.CMN'
	INCLUDE	'FERRET_CMN:FERRET.PARM'
#endif

* calling argument declarations:
	REAL*8	lo_val, hi_val
	INTEGER	idim, grid

* internal variable declarations:
	INTEGER	line, un

* simplest length
	AXIS_DISTANCE = hi_val - lo_val

* convert to uniform units
	line = grid_line(idim,grid)
	un   = line_unit_code(line)
	AXIS_DISTANCE = AXIS_DISTANCE * un_convert( un )

	RETURN
	END
