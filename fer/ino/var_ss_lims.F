	SUBROUTINE VAR_SS_LIMS( idim, cx, lo_ss, hi_ss )

* get the subscript limits for a "file variable"

* programmer - steve hankin
* NOAA/PMEL, Seattle, WA - Tropical Modeling and Analysis Program
* written for VAX computer under VMS operating system
*
* V200:  5/19/89 - created
*	10/25/89 - streamlined determining file variable number
* V300: 10/28/93 - apply limits to pseudo-variables, too, if grid is suitable
*	note: this addition was made YEARS after the logic here was developed
*	      there are uncertainties whether special circumstances in FERRET
*	      may lead to problem behavior

#ifdef unix
	include 'tmap_format/tmap_dims.parm'
	include 'tmap_format/xdset_info.cmn_text'
	external xdset_info_data
	include	'tmap_format/xtm_grid.cmn_text'
	external xgt_grid_data
	include	'ferret_cmn/ferret.parm'
	include	'ferret_cmn/errmsg.parm'
	include	'ferret_cmn/xcontext.cmn'
	include	'ferret_cmn/xfr_grid.cmn'	! 10/93
#else
	INCLUDE 'TMAP_FORMAT:TMAP_DIMS.PARM'
	INCLUDE 'TMAP_FORMAT:XDSET_INFO.CMN'
	INCLUDE	'TMAP_FORMAT:XTM_GRID.CMN'
	INCLUDE	'FERRET_CMN:FERRET.PARM'
	INCLUDE	'FERRET_CMN:ERRMSG.PARM'
	INCLUDE	'FERRET_CMN:XCONTEXT.CMN'
	include	'FERRET_CMN:XFR_GRID.CMN'
#endif

* calling argument declarations
	INTEGER idim, cx, lo_ss, hi_ss

* internal variable declarations
	INTEGER dset, cat, var, grid, line, status
	INTEGER TM_GET_GRIDNUM		! 10/93

* initialize
	dset = cx_data_set(cx)
	var  = cx_variable(cx)
	cat  = cx_category(cx)
	grid = cx_grid(cx)
	IF ( grid .NE. unspecified_int4 ) line = grid_line( idim, grid )

* pseudo-variables - added 10/93
* (must be a known axis and not be ABSTRACT)
	IF ( cat  .EQ. cat_pseudo_var
     . .AND. grid .NE. unspecified_int4
     . .AND. line .NE. unspecified_int4
     . .AND. grid .NE. TM_GET_GRIDNUM('ABSTRACT')  ) THEN
	   lo_ss = 1
	   hi_ss = line_dim( line )
	   RETURN
	ENDIF
* ... end of 10/93 additions

* cannot return answer under several conditions
	IF ( cat  .NE. cat_file_var
     .  .OR. dset .EQ. unspecified_int4
     .  .OR. line .EQ. mnormal
     .  .OR. cx_unstand_grid(cx)	 ) THEN
	   lo_ss = unspecified_int4
	   hi_ss = unspecified_int4
	   RETURN
	ENDIF

* return subscript limits
 110	lo_ss = ds_grid_start(idim, var)
	hi_ss = ds_grid_end  (idim, var)
 1000	RETURN
	END
