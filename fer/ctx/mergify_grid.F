	SUBROUTINE MERGIFY_GRID( cx, uvar )

* initialize the grid information about a uvar with information from the
* last (first merged) component in its definition

* programmer - steve hankin
* NOAA/PMEL, Seattle, WA - Tropical Modeling and Analysis Program
* written for VAX computer under VMS operating system
*
* V200:  7/11/89
*	11/29/89 - bug fixes for b grid in a=10, b=a*j[j=1:10]
*	 4/16/90 - extracted MERGIFY_GRID_AXIS greatly shortening this
* V420: 9/95	 - Support for implicit grids: deallocate after it is used
	
#ifdef unix
	include 'ferret_cmn/ferret.parm'
	include	'ferret_cmn/errmsg.parm'
	include	'ferret_cmn/interp_stack.parm'
	include	'ferret_cmn/xcontext.cmn'
	include	'ferret_cmn/xvariables.cmn'
	include	'ferret_cmn/xprog_state.cmn'
#else
	INCLUDE 'FERRET_CMN:FERRET.PARM'
	INCLUDE	'FERRET_CMN:ERRMSG.PARM'
	INCLUDE	'FERRET_CMN:INTERP_STACK.PARM'
	INCLUDE	'FERRET_CMN:XCONTEXT.CMN'
	INCLUDE	'FERRET_CMN:XVARIABLES.CMN'
	INCLUDE	'FERRET_CMN:XPROG_STATE.CMN'
#endif

* calling argument declarations
	INTEGER	cx, uvar

* local variable declarations
	LOGICAL itsa_pvar
	INTEGER	idim, cat, status

* initialize
	cat = cx_category(cx)
	itsa_pvar = cat .EQ. cat_pseudo_var

* data set dependencies
	IF ( itsa_pvar ) THEN
* ... pseudo-vars can depend on data set only through g=grid
	   cx_dset_gvn(cx) = cx_dset_gvn(cx) .OR. .NOT.cx_grid_needs_dset
	ELSE
	   cx_dset_gvn(cx) = cx_dset_gvn(cx) 
     .				.OR. ( cx_data_set(cx) .EQ. pdset_irrelevant )
	ENDIF

* loop through the axes
	DO 100 idim = 1, 4
 100	CALL MERGIFY_GRID_AXIS( cx, idim, uvar )

	RETURN
	END


