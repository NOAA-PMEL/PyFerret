	SUBROUTINE PARSE_NAM_DSET_GRD( full_var_name, default_cx,
     .				       dset, grid, cat, var, status )

* given a full variable specification of the form:
*			(eg.) TEMP[D=n,G=gname] (with optional region info)
* return context information describing variable, data set and  grid
* ... if necessary determine the grid in the process

* programmer - steve hankin
* NOAA/PMEL, Seattle, WA - Tropical Modeling and Analysis Program

* revision 0.0 - 7/27/88 - based on GET_VAR_CONTEXT and PARSE_VAR_NAME
* revision 0.1 - 3/14/89 - check to see if default data set is known
* V200:   8/9/89 - eliminated unknown data set error
* V402: 7/95 - Allow nested [] of the form vname1[G=vname2[D=dset]]
*	     - Guts of routine extracted to PARSE_NAM_DSET
	
#ifdef unix
	include	'ferret_cmn/ferret.parm'
	include	'ferret_cmn/errmsg.parm'
	include	'ferret_cmn/xcontext.cmn'
#else
	INCLUDE	'FERRET_CMN:FERRET.PARM'
	INCLUDE	'FERRET_CMN:ERRMSG.PARM'
	INCLUDE	'FERRET_CMN:XCONTEXT.CMN'
#endif

* calling argument declarations:
	INTEGER		default_cx, dset, grid, cat, var, status
	CHARACTER*(*)	full_var_name

* internal variable declarations:
	INTEGER		IGRID, mods_cx

* most of the work extracted to this routine (7/95)
	dset = unspecified_int4		! no prejudices ??? (7/95)
	CALL PARSE_NAM_DSET( full_var_name, default_cx,
     .			     dset, cat, var, mods_cx, status )

* determine the defining grid
* (although mods_cx has been deallocated nothing has over-written it yet)
	IF ( var .NE. munknown_var_name ) THEN
	   IF ( cx_unstand_grid( mods_cx ) ) THEN
	      grid = cx_grid( mods_cx )
	   ELSE
	      grid = IGRID( dset, cat, var, status )
	      IF ( status .NE. ferr_ok ) GOTO 5000
	   ENDIF
	ENDIF

* error exit
 5000	RETURN
	END
