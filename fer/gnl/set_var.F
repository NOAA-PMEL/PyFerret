	SUBROUTINE SET_VAR( vname, status )

* perform a SET VARIABLE on a variable from a data set of type EZ

* programmer - steve hankin
* NOAA/PMEL, Seattle, WA - Tropical Modeling and Analysis Program
* written for VAX computer under VMS operating system
*
* V200: 10/23/89
* 	 3/26/90 - renamed from SET_EZ_VAR.  Allowed user-defined vars
* V230:  7/23/92 - trap attempts to set units on user variable
*       10/22/92 - added units on uvars  (increased units string to 24 chars)
*        11/9/92 - made units and title case-sensitive
* V402: 7/95 - use PARSE_NAM_DSET instead of PARSE_NAM_DSET_GRD so the
*		grid is never determined and the underlying uvar defn need
*		not be based on variables that are valid at this moment
* V420: 10/95 - can apply SET VARIABLE/BAD=/UNIT=/TITLE= to GT and CDF files
*Linux Port 1/97 *kob* - Added a preprocessor include for tmap_dset.parm
*			  because it needed a preprocessor.
#ifdef unix
	include 'tmap_format/tmap_dims.parm'
#	include "tmap_format/tmap_dset.parm"
	include 'tmap_format/tmap_errors.parm'
	include 'tmap_format/xdset_info.cmn_text'
	external xdset_info_data
	include 'ferret_cmn/ferret.parm'
	include 'ferret_cmn/errmsg.parm'
	include 'ferret_cmn/slash.parm'
	include 'ferret_cmn/xprog_state.cmn'
	include 'ferret_cmn/xvariables.cmn'
	include 'ferret_cmn/xrisc.cmn'
#else
	INCLUDE 'TMAP_FORMAT:TMAP_DIMS.PARM'
	INCLUDE 'TMAP_FORMAT:TMAP_DSET.PARM'
	INCLUDE 'TMAP_FORMAT:TMAP_ERRORS.PARM'
	INCLUDE 'TMAP_FORMAT:XDSET_INFO.CMN'
	INCLUDE 'FERRET_CMN:FERRET.PARM'
	INCLUDE 'FERRET_CMN:ERRMSG.PARM'
	INCLUDE 'FERRET_CMN:SLASH.PARM'
	INCLUDE 'FERRET_CMN:XPROG_STATE.CMN'
	INCLUDE 'FERRET_CMN:XVARIABLES.CMN'
	INCLUDE 'FERRET_CMN:XRISC.CMN'
#endif

* calling argument declarations
	INTEGER		status
	CHARACTER	vname*(*)

* internal variable declarations:
	LOGICAL		VALID_GRID, have_mods, ez
	INTEGER		GRID_FROM_NAME, mod_cx, 
     .			dset, grid, cat, var, loc, TMAP_status
	REAL		bad
	CHARACTER	grid_name*16, title*128, units*24

* decode specified variable name of form VNAME[D=dset]
	dset = unspecified_int4			! default is global variable
	CALL PARSE_NAM_DSET( vname, cx_last, dset,
     .			     cat, var, mod_cx, status )
* ... RISC FORTRAN bug prevents "//" on char*(*) variable vname
	risc_buff = vname
	len_rbuff = LEN(vname)
	IF ( status .NE. ferr_ok ) RETURN
	IF ( var .EQ. munknown_var_name ) GOTO 5005

* must be an EZ or user-defined variable
	IF ( cat .EQ. cat_file_var ) THEN
	   ez = ds_type(dset) .EQ. pds_type_ez
!	   IF ( .NOT. ez ) GOTO 5010		! removed restriction 10/95
	ELSEIF ( cat .EQ. cat_user_var ) THEN   ! uvars bad, too, until 10/92
           ez = .FALSE.
	ELSE
	   GOTO 5015
	ENDIF

* process qualifiers given
	loc = qual_given( slash_set_var_title )
	IF ( loc .GT. 0 ) THEN
	   have_mods = .TRUE.
           CALL EQUAL_STR_LC( cmnd_buff(qual_start(loc):qual_end(loc)),
     .			      title, status )
           IF ( status .NE. ferr_ok ) RETURN
	ELSE
	   title = char_init80
	ENDIF
	loc = qual_given( slash_set_var_unit )
	IF ( loc .GT. 0 ) THEN
	   have_mods = .TRUE.
           CALL EQUAL_STR_LC( cmnd_buff(qual_start(loc):qual_end(loc)),
     .			      units, status )
           IF ( status .NE. ferr_ok ) RETURN
	ELSE
	   units = char_init16
	ENDIF
	loc = qual_given( slash_set_var_grid )
	IF ( loc .GT. 0 ) THEN
	   have_mods = .TRUE.
           CALL EQUAL_STRING( cmnd_buff(qual_start(loc):qual_end(loc)),
     .			      grid_name, status )
           IF ( status .NE. ferr_ok ) RETURN
	   grid = GRID_FROM_NAME( grid_name, cx_last, status )
           IF ( status .NE. ferr_ok ) RETURN
	   IF ( .NOT.VALID_GRID( grid ) ) GOTO 5500
	ELSE
	   grid = int4_init
	ENDIF
	loc = qual_given( slash_set_var_bad )
	IF ( loc .GT. 0 ) THEN
	   have_mods = .TRUE.
	   CALL EQUAL_VAL( cmnd_buff(qual_start(loc):qual_end(loc)),
     .			   bad, status )
	   IF ( status .NE. ferr_ok ) RETURN
	ELSE
	   bad = real4_init
	ENDIF

* note on EZ vars:
* modifying the variable info invalidates data in memory
*        the purging of memory variables below is unnecessarily strict in both
*	 the uvar and fvar cases.  FVAR's can be modified in memory to change
*	 mr_grid, units, etc.  They need to be deleted only if they were on 
* 	 non-standard grids or units.  The UVAR's need to be purged only if
*	 they depend on the FVAR being changed)
	IF ( ez ) THEN
	   IF ( have_mods ) THEN
	      CALL PURGE_FILE_VAR( var ) ! remove memory-resident variables
	      CALL PURGE_ALL_UVARS	 ! delete (possibly dependent) uvars
	      CALL EZ_MOD_VARS( dset,var,title,units,grid,bad,tmap_status )
	      IF ( tmap_status .NE. merr_ok ) THEN
	         status = ferr_TMAP_error
	         RETURN
	      ENDIF
	   ENDIF
	ELSEIF ( cat .EQ. cat_file_var ) THEN
* ... other file variables (10/95)
	   IF ( grid .NE. int4_init   ) GOTO 5112
	   IF ( bad  .NE. real4_init  ) THEN
	      CALL PURGE_FILE_VAR( var ) ! remove memory-resident variables
	      CALL PURGE_ALL_UVARS	 ! delete (possibly dependent) uvars
	      ds_missing_flag(var) = bad
	   ENDIF
	   IF ( units .NE. char_init16 ) ds_var_units( var ) = units
	   IF ( title .NE. char_init80 ) ds_var_title( var ) = title
	ELSE
* ... user-defined var
	   IF ( grid .NE. int4_init   ) GOTO 5110
	   IF ( bad  .NE. real4_init  ) GOTO 5115
	   IF ( units .NE. char_init16 ) uvar_units( var ) = units
	   IF ( title .NE. char_init80 ) uvar_title( var ) = title
	ENDIF

* successful completion
	status = ferr_ok
	RETURN

* error exits
 5000	RETURN
 5005	CALL ERRMSG( ferr_unknown_variable, status,
     .		    '(data set ok ?): '//risc_buff(:len_rbuff), *5000 )
! 5010	CALL ERRMSG( ferr_invalid_command, status,
!     .		risc_buff(:len_rbuff)//' is from a self-documented data set'
!     .		//pCR//'Can only SET VARIABLE on variable from an EZ data set',
!     .							 *5000 )
 5015	CALL ERRMSG( ferr_invalid_command, status,
     .		'Cannot SET VARIABLE on '//risc_buff(:len_rbuff)//pCR//
     .		'Only on variables from "EZ" data sets',
     .							 *5000 )
!     .		'Only on user-defined and variables from "EZ" data sets',
 5100	CALL ERRMSG( ferr_erreq, status,
     .		cmnd_buff(qual_start(loc):qual_end(loc)), *5000 )
 5110   CALL ERRMSG( ferr_invalid_command, status,
     .               'cannot specify grid for '//risc_buff(:len_rbuff)//pCR//
     .               'Grid for a user-defined variable is implied '//
     .               'by its definition', *5000 )
 5112   CALL ERRMSG( ferr_invalid_command, status,
     .               'cannot specify grid for '//risc_buff(:len_rbuff),
     .               						*5000 )
 5115   CALL ERRMSG( ferr_invalid_command, status,
     .		     'Cannot specify bad value for '//risc_buff(:len_rbuff),
     .							*5000 )
 5500	CALL ERRMSG( ferr_grid_definition, status, grid_name, *5000 )
	END
