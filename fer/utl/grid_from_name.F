	INTEGER FUNCTION GRID_FROM_NAME(lc_name, dflt_cx, status)

* given a name which may be the name of a grid or the name of a variable
* with possible data set and grid name modifiers return a grid number

* programmer - steve hankin
* NOAA/PMEL, Seattle, WA - Tropical Modeling and Analysis Program
* written for VAX computer under VMS operating system
*
* revision 0.0 - 7/29/88
* V200:  4/27/90 - bug fix: status not set on successful return
* Unix/RISC port - 11/1/91: bug fix: status not checked returning from
*                           PARSE_NAM_DSET_GRD.  "SHOW GRID T" blew up.
* V300: 2/93 - return unspecified if error exit (caught by f77 3.2)
* V491+ 7/98 *sh* - allow case-sensitive variable names in single quotes 

#ifdef unix
	include	'ferret_cmn/ferret.parm'
	include	'ferret_cmn/errmsg.parm'
#else
	INCLUDE	'FERRET_CMN:FERRET.PARM'
	INCLUDE	'FERRET_CMN:ERRMSG.PARM'
#endif

* calling argument declarations:
	INTEGER		dflt_cx, status
	CHARACTER*(*)	lc_name

* local variable declarations:
	INTEGER		STR_UPCASE, TM_GET_GRIDNUM, TM_LENSTR1,
     .			vax_code, iset, grid, category, variable
	CHARACTER	name*32

* change it to upper case
	vax_code = STR_UPCASE( name, lc_name )

* try to identify it first as a grid name
	grid = TM_GET_GRIDNUM( name )
	IF ( grid .NE. unspecified_int4 ) GOTO 1000
	
* well then, identify the variable named and get its grid
! 7/98	CALL PARSE_NAM_DSET_GRD( name, dflt_cx,
	CALL PARSE_NAM_DSET_GRD( lc_name, dflt_cx,
     .				iset, grid, category, variable, status )
        IF ( status .NE. ferr_ok ) GOTO 5000       ! 11/91 *sh*
	IF ( variable .EQ. munknown_var_name ) CALL ERRMSG
     .       ( ferr_unknown_grid, status,
     .             lc_name(:TM_LENSTR1(lc_name)),*5000 )
	
* success
 1000	GRID_FROM_NAME = grid
	status = ferr_ok
	RETURN

* error exit(s)
 5000	GRID_FROM_NAME = unspecified_int4     ! added 2/93
  	RETURN

	END
