	INTEGER FUNCTION KNOWN_GRID ( data_set, category, variable )

* find the defining grid for the given variable
* if unknown - return unknown flag

* programmer - steve hankin
* NOAA/PMEL, Seattle, WA - Tropical Modeling and Analysis Program
* written for VAX computer under VMS operating system
*
* V200:   7/6/89 - based on IGRID
* V230:  7/21/92 - check for unspecified data set on calc_var

#ifdef unix
	include	'tmap_format/tmap_dims.parm'
	include	'tmap_format/xdset_info.cmn_text'
	external xdset_info_data
	include	'ferret_cmn/ferret.parm'
	include	'ferret_cmn/xvariables.cmn'
	include	'ferret_cmn/xcalc_vars.cmn'
#else
	INCLUDE	'TMAP_FORMAT:TMAP_DIMS.PARM'
	INCLUDE	'TMAP_FORMAT:XDSET_INFO.CMN'
	INCLUDE	'FERRET_CMN:FERRET.PARM'
	INCLUDE	'FERRET_CMN:XVARIABLES.CMN'
	INCLUDE	'FERRET_CMN:XCALC_VARS.CMN'
#endif

* calling argument declarations:
	INTEGER	data_set, category, variable

* internal variable declarations:
	INTEGER	dset

* different procedure for determining grid from each category
	IF     ( category .EQ. cat_file_var ) THEN

	   KNOWN_GRID = ds_grid_number( variable )


	ELSEIF ( category .EQ. cat_calc_var ) THEN

           IF ( data_set .EQ. unspecified_int4 ) THEN
              KNOWN_GRID = unspecified_int4
           ELSE
	      KNOWN_GRID = cvar_grid( variable, data_set )
           ENDIF


	ELSEIF ( category .EQ. cat_user_var ) THEN

* ... allow "abstract" expressions that don't depend on data set
	   IF ( .NOT.uvar_need_dset(variable)
     .	   .OR. data_set .EQ. unspecified_int4 ) THEN
	      dset = pdset_irrelevant
	   ELSE
	      dset = data_set
	   ENDIF
	   KNOWN_GRID = uvar_grid( variable, dset )



	ELSEIF ( category .EQ. cat_temp_var ) THEN

	   KNOWN_GRID = mr_grid( variable )



	ELSE

	   STOP 'KNOWN_GR'

	ENDIF

	RETURN
	END
