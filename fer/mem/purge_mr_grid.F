	SUBROUTINE PURGE_MR_GRID( grid, status )

* eliminate all stored mrs on the given grid

* programmer - steve hankin
* NOAA/PMEL, Seattle, WA - Tropical Modeling and Analysis Program
* written for VAX computer under VMS operating system
*
* V200:  8/4/89
*	12/6/89 - purge mgrid_abstract_last, too
* Unix/RISC port - 4/25/91 *sh*: longer ds_name
*Linux Port 1/97 *kob* - Added a preprocessor include for tmap_dset.parm
*			  because it needed a preprocessor.
#ifdef unix
	include	'tmap_format/tmap_dims.parm'
#	include	"tmap_format/tmap_dset.parm"
	include	'tmap_format/tmap_errors.parm'
	include 'tmap_format/xdset_info.cmn_text'
	external xdset_info_data
	include 'tmap_format/xtm_grid.cmn_text'
	external xgt_grid_data
	include	'ferret_cmn/ferret.parm'
	include	'ferret_cmn/errmsg.parm'
#else
	INCLUDE	'TMAP_FORMAT:TMAP_DIMS.PARM'
	INCLUDE	'TMAP_FORMAT:TMAP_DSET.PARM'
	INCLUDE	'TMAP_FORMAT:TMAP_ERRORS.PARM'
	INCLUDE 'TMAP_FORMAT:XDSET_INFO.CMN'
	INCLUDE 'TMAP_FORMAT:XTM_GRID.CMN'
	INCLUDE	'FERRET_CMN:FERRET.PARM'
	INCLUDE	'FERRET_CMN:ERRMSG.PARM'
#endif
!	INCLUDE 'FERRET_CMN:XPROG_STATE.CMN'
#ifdef unix
	include 'ferret_cmn/xvariables.cmn'
	include 'ferret_cmn/xcalc_vars.cmn'
#else
	INCLUDE 'FERRET_CMN:XVARIABLES.CMN'
	INCLUDE 'FERRET_CMN:XCALC_VARS.CMN'
#endif
!	INCLUDE 'FERRET_CMN:XFR_GRID.CMN'

* calling argument declaration:
	INTEGER	grid, status

* internal variable declarations:
	INTEGER	TM_LENSTR1, mr, ivar, cvar, dset, glen

* the lowest grids are protected
	IF ( grid .LE. 6 ) GOTO 5200

* make sure grid isn't used by any file variables
	DO 10 ivar = 1, maxvars
	   dset = ds_var_setnum( ivar )
	   IF ( dset .EQ. set_not_open ) GOTO 10
	   IF ( ds_grid_number(ivar) .EQ. grid ) GOTO 5100
 10	CONTINUE

* make sure it isn't implicitly used by a cvar that depends on file variables
	DO 25 dset = 1, maxdsets
	   IF ( ds_name(dset) .EQ. char_init40 ) GOTO 25
	   DO 20 cvar = 1, num_calc_vars
 20	   IF ( cvar_grid(cvar,dset) .EQ. grid ) GOTO 5100
 25	CONTINUE

* the coast is clear
* ...clear out the memory variables
	DO 100 mr = 1, max_mrs
	   IF ( mr_protected( mr ) .EQ. mr_deleted ) GOTO 100
	   IF ( mr_grid ( mr )     .NE. grid	   ) GOTO 100
	   CALL DELETE_VARIABLE( mr )
 100	CONTINUE

* and wipe out all user-defined variables (they may depend on it indirectly)
	CALL PURGE_ALL_UVARS

* success
	status = ferr_ok
	RETURN

* error exit
 5000	RETURN
 5100	glen = TM_LENSTR1( grid_name( grid ) )
	CALL ERRMSG( ferr_grid_definition, status,
     .		'grid '//grid_name(grid)(:glen)//' used by data set '
     .		//ds_name(dset), *5000 )
 5200	CALL ERRMSG( ferr_grid_definition, status,
     .		'protected grid: '//grid_name(grid), *5000 )
	END

