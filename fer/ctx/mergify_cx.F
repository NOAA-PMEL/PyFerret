	SUBROUTINE MERGIFY_CX( dflt_cx, uvar, cx, status )

* impose a "merged look" on context cx by setting the data set to the default,
* an dsetting the grid to the merged grid

* programmer - steve hankin
* NOAA/PMEL, Seattle, WA - Tropical Modeling and Analysis Program
* written for VAX computer under VMS operating system
*
* V200:  6/20/89
	
#ifdef unix
	include 'ferret_cmn/ferret.parm'
	include 'ferret_cmn/errmsg.parm'
	include	'ferret_cmn/xcontext.cmn'
	include	'ferret_cmn/xvariables.cmn'
	include	'ferret_cmn/xfr_grid.cmn'
#else
	INCLUDE 'FERRET_CMN:FERRET.PARM'
	INCLUDE 'FERRET_CMN:ERRMSG.PARM'
	INCLUDE	'FERRET_CMN:XCONTEXT.CMN'
	INCLUDE	'FERRET_CMN:XVARIABLES.CMN'
	INCLUDE	'FERRET_CMN:XFR_GRID.CMN'
#endif

* calling argument declarations
	INTEGER	dflt_cx, uvar, cx, status

* local variable declarations
	INTEGER	KNOWN_GRID, idim

* impose default data set or no data set as appropriate
	IF ( uvar_need_dset( uvar ) ) THEN
	   cx_data_set( cx ) = cx_data_set( dflt_cx )
	ELSE
	   cx_data_set( cx ) = pdset_irrelevant
	ENDIF

* point context to this user variable
	cx_variable( cx ) = uvar
	cx_category( cx ) = cat_user_var

* impose merged grid
	cx_grid( cx ) = KNOWN_GRID( cx_data_set(cx), cat_user_var, uvar )
	cx_unstand_grid( cx ) = .FALSE.

* wipe out any transformations from first component of merged context
	DO 100 idim = 1, 4
	   cx_trans    ( idim, cx ) = trans_no_transform
	   cx_trans_arg( idim, cx ) = bad_val4
 100	CONTINUE

	RETURN

* error exit
! 5000	RETURN
	END
