	CHARACTER*(*) FUNCTION GNAME_FROM_NUMBER( dset, grid, len )

* given a grid number return the name of the first file variable in this dset
* that is defined on this grid.  If the variable is from a different
* data set than the one passed in the call qualify the variable name with
* [D=name]
* If no variables are defined on this grid then return the name of the grid
* itself.

* programmer - steve hankin
* NOAA/PMEL, Seattle, WA - Tropical Modeling and Analysis Program
* written for VAX computer under VMS operating system
*
* revision 0.0 - 8/1/88
*Linux Port 1/97 *kob* - Added a preprocessor include for tmap_dset.parm
*			  because it needed a preprocessor.

#ifdef unix
	include 'tmap_format/tmap_dims.parm'
#	include "tmap_format/tmap_dset.parm"
	include 'tmap_format/xdset_info.cmn_text'
	external xdset_info_data
	include	'tmap_format/xtm_grid.cmn_text'
	external xgt_grid_data
	include	'ferret_cmn/ferret.parm'
	include	'ferret_cmn/errmsg.parm'
#else
	INCLUDE 'TMAP_FORMAT:TMAP_DIMS.PARM'
	INCLUDE 'TMAP_FORMAT:TMAP_DSET.PARM'
	INCLUDE 'TMAP_FORMAT:XDSET_INFO.CMN'
	INCLUDE	'TMAP_FORMAT:XTM_GRID.CMN'
	INCLUDE	'FERRET_CMN:FERRET.PARM'
	INCLUDE	'FERRET_CMN:ERRMSG.PARM'
#endif

* calling argument declarations:
	INTEGER		dset, grid, len

* local variable declarations:
	LOGICAL	in_set
	INTEGER	TM_LENSTR1, var, dset2

* search for a file variable on both the right grid and data set
	DO 100 var = 1, maxvars
	   IF ( ds_var_setnum (var) .EQ. set_not_open ) GOTO 100

	   IF ( ds_grid_number(var) .EQ. grid
     .	  .AND. ds_var_setnum (var) .EQ. dset ) GOTO 400

 100	CONTINUE

* no luck - search for a file variable on the right grid in another data set
	DO 200 var = 1, maxvars
	   IF ( ds_var_setnum (var) .EQ. set_not_open ) GOTO 200

	   IF ( ds_grid_number(var) .EQ. grid ) GOTO 300

 200	CONTINUE

		         
* no luck again - no file variables are defined on this grid - return grid name
	GNAME_FROM_NUMBER = grid_name( grid )
	GOTO 500

* right grid but another data set
 300	dset2 = ds_var_setnum(var)
	GNAME_FROM_NUMBER = 
     .           ds_var_code(var)(:TM_LENSTR1(ds_var_code(var)))//
     .		'[D='//ds_name(dset2)(:TM_LENSTR1(ds_name(dset2)))//']'
	GOTO 500

* right grid and variable
 400	GNAME_FROM_NUMBER = ds_var_code(var)

* measure the string	   
 500	len = TM_LENSTR1( GNAME_FROM_NUMBER )
	RETURN
	END
