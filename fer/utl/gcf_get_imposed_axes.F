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
* V491: 1/98 - make test function TIMES2 impose its T axis


* calling argument declarations
	INTEGER	gcfcn, cx_base, axes(4), status

* internal variable declarations
	LOGICAL known_limits
	INTEGER	CGRID_AXIS, TM_LENSTR1, CX_DIM_LEN, CAXIS_LEN,
     .		cx, idim, axis_will_be(4), n, grid, modulo
	REAL*8	TM_WORLD, dlo, dhi, del
	CHARACTER*40 GCF_NAME, buff, unit

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
	include 'ferret_cmn/xcontext.cmn'       ! for TIMES2
	include 'ferret_cmn/xvariables.cmn'     ! for TIMES2
#else
	INCLUDE 'TMAP_FORMAT:TMAP_DIMS.PARM'
	INCLUDE 'TMAP_FORMAT:XTM_GRID.CMN_TEXT'
	INCLUDE 'FERRET_CMN:FERRET.PARM'
	INCLUDE 'FERRET_CMN:ERRMSG.PARM'
	INCLUDE 'FERRET_CMN:GRID_CHG_FCNS.PARM'
	INCLUDE 'FERRET_CMN:XGRID_CHG_FCNS.CMN'
	INCLUDE 'FERRET_CMN:XTEXT_INFO.CMN'	! W/ WW_DIM_NAME
	INCLUDE 'FERRET_CMN:XFR_GRID.CMN'       ! w/ mgrid_abstract
	INCLUDE 'FERRET_CMN:XCONTEXT.CMN'       ! for TIMES2
	INCLUDE 'FERRET_CMN:XVARIABLES.CMN'     ! for TIMES2
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

* process what axes we can: either "merged" (determined by grid stack logic),
*	NORMAL, and ABSTRACT ... but not IMPOSED ("supplied herein")
	DO 100 idim = 1, 4
	   IF (axis_will_be(idim) .EQ. pgc_axis_is_abstract) THEN
	      axes(idim) = grid_line(1,mgrid_abstract)
	   ELSEIF (axis_will_be(idim) .EQ. pgc_axis_is_normal) THEN
	      axes(idim) = mnormal
	   ELSEIF (axis_will_be(idim) .EQ. pgc_axis_implied_by_args) 
     .             THEN
	      axes(idim) = unspecified_int4   ! ... determined elsewhere
           ELSEIF (axis_will_be(idim) .EQ. pgc_axis_supplied_herein) 
     .             THEN
* nothing to do here
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

	IF ( gcfcn .EQ. p_reshape ) THEN
* *** RESHAPE function ***
* The second argument contains the desired grid. (It cannot simply be merged,
* however, because the result limits have no obvious relation to the inputs)
	   cx = cx_base 	! component 2: the reshaping grid
	   DO 210 idim = 1, 4
	      axes(idim) = CGRID_AXIS ( idim, cx )
 210	   CONTINUE



	ELSEIF ( gcfcn .EQ. p_times2 ) THEN
* *** TIMES2 (debugging) function ***
	   cx = cx_base 	! the argument context

	   IF ( CGRID_AXIS(t_dim,cx) .EQ. mnormal
     .	   .OR. CGRID_AXIS(t_dim,cx) .EQ. unspecified_int4 )
     .		CALL ERRMSG(ferr_unknown_arg, status,
     .			'arg of TIMES2(v) must have a T axis',*5000)

	   grid = cx_grid(cx)

* Note: This routine is called during the "getgrid" pass of Ferret.
* During the "getgrid" pass the context limits of the argument are not reliably
* available. For example, in "TIMES2(V)" the limits of V will appear
* to be cx_lo/hi_ss=1:1. Even in the case of TIMES2(V[L=1:4]) at this
* stage it will appear that V has limits L=1:4 but the actual arg
* that will be obtained later may differ (example: the definition of V may be
* LET V = V2[L=5:10]). The best that we can do about this situation is
* 1. check cx_given to determine if the limits were explicit. If not,
*	then use the limits of the full axis
* 2. in GCF_IMPOSE_RESULT_LIMS and again in the function, itself, double
*	check the axis we have defined against the actual argument recieved.
*	If the axis is inappropriate then flag an error and coach the user
*	that the argument limits are ambiguous or contradictory.

* compute the lo/hi/delta of the new custom (regular) axis

* are the argument's limits along this axis known ?
	known_limits = cx_given(t_dim, cx)	! given right in fcn arg list
	IF ( cx_category(cx) .EQ. cat_user_var ) THEN
* ... user variables may contain limits embedded in its definition
	   known_limits = known_limits .OR.
     .          uvar_given(t_dim,cx_variable(cx))
     .          .LE. uvlim_gvn_xact
	ENDIF
        IF ( known_limits ) THEN
* ... limits of arg are explicit -- use 'em

           n = CX_DIM_LEN(t_dim, cx)
           dlo = TM_WORLD(cx_lo_ss(cx,t_dim), cx_grid(cx), t_dim,
     .          box_middle )
           dhi = TM_WORLD(cx_hi_ss(cx,t_dim), cx_grid(cx), t_dim,
     .          box_middle )
	   ELSE
* ... limits of arg are not explicit -- use axis lims (and hope for the best!)
	      n = CAXIS_LEN(t_dim, cx)
	      dlo = TM_WORLD(1, cx_grid(cx), t_dim, box_middle )
	      dhi = TM_WORLD(n, cx_grid(cx), t_dim, box_middle )
	   ENDIF
	   IF ( n .EQ. 1 ) THEN
	      del = 1.D0
	   ELSE
	      del = (dhi-dlo)/(n-1)
	   ENDIF

*	Create and set the custom axis
* NOTE: It is important that **somewhere** inside of Ferret TM_MAKE_DYN_LINE
*	is called -- else it will not be linked in and available to the
*	external functions 
* Use TM_GET_LIKE_DYN_LINE to create axes which are children of an argument ax
	   CALL TM_MAKE_DYN_LINE(t_dim, dlo, dhi, del,
     .                                  'Hertz', .FALSE.,
     .                                  axes(t_dim), status)
	   IF (status .NE. ferr_ok) GOTO 5000


**************
*     FOR EXTERNAL FUNCTIONS


	ELSEIF (gcfcn .GT. gfcn_num_internal ) THEN

           IF (axis_will_be(1) .EQ. pgc_axis_supplied_herein
     .          .OR. axis_will_be(2) .EQ. pgc_axis_supplied_herein
     .          .OR. axis_will_be(3) .EQ. pgc_axis_supplied_herein
     .          .OR. axis_will_be(4) .EQ. pgc_axis_supplied_herein) THEN
              CALL EFCN_GET_CUSTOM_AXES(gcfcn, cx_base)
           ENDIF

           DO 400 idim = 1, 4

              IF (axis_will_be(idim) .EQ. pgc_axis_supplied_herein) THEN
                 CALL EFCN_GET_CUSTOM_AXIS(gcfcn, idim, dlo, dhi, del,
     .                unit, modulo)
                 CALL TM_MAKE_DYN_LINE(idim, dlo, dhi, del,
     .                unit, modulo, axes(idim), status)
                 IF (status .NE. ferr_ok) GOTO 5000
              ENDIF


 400       CONTINUE

	ENDIF

* successful completion
	status = ferr_ok
	RETURN

* error exit
 5000	RETURN
	END
