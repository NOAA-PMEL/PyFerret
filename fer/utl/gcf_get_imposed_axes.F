	SUBROUTINE GCF_GET_IMPOSED_AXES( gcfcn, cx_base, axes, status )

* supply the axes for those axes of the grid changing function which
* are "imposed" by the function

* the addition (or major modification) of an internal grid-changing function
* requires changes to these routines:
*	grid_chg_fcns.parm	- parameters to number the functions
*	XGRID_CHG_FCNS_DATA	- description of the function and args
*	DO_INTERNAL_GC_FCN	- code to execute the function
* (opt)	GCF_GET_IMPOSED_AXES	- return the "imposed" axes, if any
* (opt)	GCF_IMPOSE_RESULT_LIMS	- fill in limits of "imposed" axes, if any

* V450: 1/97
*	7/97 - slight reorg for external functions

* calling argument declarations
	INTEGER	gcfcn, cx_base, axes(4), status

* internal variable declarations
	LOGICAL has_imposed_axes
	INTEGER	CGRID_AXIS, TM_LENSTR1, cx, idim,
     .		axis_will_be(4)
	CHARACTER*40 GCF_NAME, buff

* Upon entry to this routine the context stack should be organized as
*	 	  default context (for this uvar)
*	 big_cx:  the merged grid context
* -->	cx_base:  argument N context
*		  argument N-1 context
*		  ...
*		  argument 1 context

#ifdef unix
	include 'tmap_format/tmap_dims.parm'
	include 'tmap_format/xtm_grid.cmn_text'
	include 'ferret_cmn/ferret.parm'
	include 'ferret_cmn/errmsg.parm'
	include 'ferret_cmn/grid_chg_fcns.parm'
	include 'ferret_cmn/xgrid_chg_fcns.cmn'
	include 'ferret_cmn/xtext_info.cmn'	! w/ ww_dim_name
	include 'ferret_cmn/xfr_grid.cmn'       ! w/ mgrid_abstract
#else
	INCLUDE 'TMAP_FORMAT:TMAP_DIMS.PARM'
	INCLUDE 'TMAP_FORMAT:XTM_GRID.CMN_TEXT'
	INCLUDE 'FERRET_CMN:FERRET.PARM'
	INCLUDE 'FERRET_CMN:ERRMSG.PARM'
	INCLUDE 'FERRET_CMN:GRID_CHG_FCNS.PARM'
	INCLUDE 'FERRET_CMN:XGRID_CHG_FCNS.CMN'
	INCLUDE 'FERRET_CMN:XTEXT_INFO.CMN'	! W/ WW_DIM_NAME
	INCLUDE 'FERRET_CMN:XFR_GRID.CMN'       ! w/ mgrid_abstract
#endif

* initialize
* ... consolidate info about function into local array to blur distinctions
*     between internal and external functions
	IF ( gcfcn .LE. gfcn_num_internal ) THEN
	   DO 10 idim = 1, 4
 10	   axis_will_be(idim) = gfcn_axis_will_be(idim,gcfcn)
	ELSE
	   call efcn_get_axis_will_be(gcfcn, axis_will_be)
	ENDIF
	has_imposed_axes = .FALSE. ! innocent until proven guilty

* process what axes we can: either "merged" (determined by grid stack logic),
*	NORMAL, and ABSTRACT ... but not IMPOSED ("supplied herein")
	DO 100 idim = 1, 4
	   IF (axis_will_be(idim) .EQ. pgc_axis_is_abstract) THEN
	      axes(idim) = grid_line(1,mgrid_abstract)
	   ELSEIF (axis_will_be(idim) .EQ. pgc_axis_is_normal) THEN
	      axes(idim) = mnormal
	   ELSEIF (axis_will_be(idim) .EQ. pgc_axis_implied_by_args) THEN
	      axes(idim) = unspecified_int4   ! ... determined elsewhere
	   ELSEIF (axis_will_be(idim) .EQ. pgc_axis_supplied_herein) THEN
	      has_imposed_axes = .TRUE.
	   ELSE	      
	      buff = GCF_NAME( gcfcn )
	      CALL ERRMSG(ferr_internal, status,
     .			'illegal axis merging parameter: '
     .			//ww_dim_name(idim)//' axis  of '
     .			//buff(:TM_LENSTR1(buff)), *5000 )
	   ENDIF
 100	CONTINUE

* *** UNRAVEL function ***
* The result of this function is a grid of ABSTRACT NORMAL NORMAL NORMAL
*   .... 7/97 changes -- handled entirely within "typical" cases above

* *** RESHAPE function ***
* The second argument contains the desired grid. (It cannot simply be merged,
* however, because the result limits have no obvious relation to the inputs)
	IF ( gcfcn .EQ. p_reshape ) THEN
	   cx = cx_base 	! component 2: the reshaping grid
	   DO 210 idim = 1, 4
	      axes(idim) = CGRID_AXIS ( idim, cx )
 210	   CONTINUE

	ELSEIF (gcfcn .GT. gfcn_num_internal ) THEN

	   IF ( has_imposed_axes ) THEN
	      STOP 'need to request externally imposed axis'
	   ENDIF

	ENDIF

* successful completion
	status = ferr_ok
	RETURN

* error exit
 5000	RETURN
	END
