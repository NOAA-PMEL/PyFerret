	SUBROUTINE DEALLO_UVAR_GRIDS( uvar )

* deallocate all dynamic grids associated with this user variable
* (clearing any memory references to this grid)

* programmer - steve hankin
* NOAA/PMEL, Seattle, WA - Tropical Modeling and Analysis Program
*
* V420 9/8/95 - new

#ifdef unix
	include 'tmap_format/tmap_dims.parm'
	include	'tmap_format/xtm_grid.cmn_text'
	external xgt_grid_data
	include 'ferret_cmn/ferret.parm'
	include 'ferret_cmn/xvariables.cmn'
#else
	INCLUDE 'TMAP_FORMAT:TMAP_DIMS.PARM'
	INCLUDE	'TMAP_FORMAT:XTM_GRID.CMN'
	INCLUDE 'FERRET_CMN:FERRET.PARM'
	INCLUDE 'FERRET_CMN:XVARIABLES.CMN'
#endif

* calling argument declarations
	INTEGER	uvar

* local declarations
	INTEGER i


* loop through all of the grids that may be associated with this
* note: could be more efficient:  check to see if this is a data set
*	dependent variable (or "EX#n") and delete only the single data set grid

	DO 100 i = 0, max_gfdl_dsets

* if the previous grid was dynamic then remove the grid (and possibly
* all memory resident variables that depend on it)
	   IF ( uvar_grid(uvar,i) .GT. max_grids ) THEN
	      CALL DEALLO_DYN_GRID(uvar_grid(uvar,i))
	   ENDIF

* set the grid pointer to its initialized state
	uvar_grid( uvar, i ) = unspecified_int4

 100	CONTINUE

	RETURN
	END
