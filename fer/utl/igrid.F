	INTEGER FUNCTION IGRID ( data_set, category, variable, status )

* find the defining grid for the given variable
* if unknown - go get it

* programmer - steve hankin
* NOAA/PMEL, Seattle, WA - Tropical Modeling and Analysis Program
* written for VAX computer under VMS operating system
*
* revision 0.0 - 4/23/86
* revision 0.6 - 11/22/88- new TM IO library var_gridber --> ds_grid_number
* V200:   7/6/89 - most guts removed to KNOWN_GRID
*	 8/29/89 - trap pseudo-variable grid requests
*	 3/26/90 - improved error messages
* V230:  7/21/92 - improved error message for unknown GFDL diagnostic var

#ifdef unix
	include	'ferret_cmn/ferret.parm'
	include	'ferret_cmn/errmsg.parm'
	include	'ferret_cmn/xvariables.cmn'
	include	'ferret_cmn/xalgebra.cmn'
#else
	INCLUDE	'FERRET_CMN:FERRET.PARM'
	INCLUDE	'FERRET_CMN:ERRMSG.PARM'
	INCLUDE	'FERRET_CMN:XVARIABLES.CMN'
	INCLUDE	'FERRET_CMN:XALGEBRA.CMN'
#endif

* calling argument declarations:
	INTEGER	data_set, category, variable, status

* internal variable declarations:
	INTEGER	KNOWN_GRID, TM_LENSTR1, dset

* only user variables have a status issue
	status = ferr_ok

* check for pseudo-variable
	IF ( category .EQ. cat_pseudo_var ) CALL ERRMSG
     .		( ferr_invalid_command, status,	'pseudo-variable '
     .		//alg_pvar(variable)(:TM_LENSTR1(alg_pvar(variable)))
     .		//' cannot be used this way',
     .		*5000 )

	IGRID = KNOWN_GRID( data_set, category, variable )

	IF ( IGRID .NE. unspecified_int4 ) RETURN

* grid unknown
	IF ( category .EQ. cat_calc_var ) THEN
	   CALL ERRMSG( ferr_invalid_command, status,
     .		'unknown data set for GFDL diagnostic variable',
     .          *5000 )
	ELSEIF ( category .NE. cat_user_var ) THEN
	   CALL ERRMSG( ferr_internal, status, 'IGRID', *5000 )
	ENDIF

* ... allow "abstract" expressions that don't depend on data set
	IF ( .NOT.uvar_need_dset(variable)
     .	   .OR. data_set .EQ. unspecified_int4 ) THEN
	   dset = pdset_irrelevant
	ELSE
	   dset = data_set
	ENDIF

	CALL GET_UVAR_GRID( variable, dset, status )
	IGRID = uvar_grid( variable, dset )

	RETURN

* error exit
 5000	RETURN
	END
