      SUBROUTINE GCF_GET_IMPOSED_AXES( gcfcn, cx_base, axes, status )

*  This software was developed by the Thermal Modeling and Analysis
*  Project(TMAP) of the National Oceanographic and Atmospheric
*  Administration's (NOAA) Pacific Marine Environmental Lab(PMEL),
*  hereafter referred to as NOAA/PMEL/TMAP.
*
*  Access and use of this software shall impose the following
*  obligations and understandings on the user. The user is granted the
*  right, without any fee or cost, to use, copy, modify, alter, enhance
*  and distribute this software, and any derivative works thereof, and
*  its supporting documentation for any purpose whatsoever, provided
*  that this entire notice appears in all copies of the software,
*  derivative works and supporting documentation.  Further, the user
*  agrees to credit NOAA/PMEL/TMAP in any publications that result from
*  the use of this software or in any product that includes this
*  software. The names TMAP, NOAA and/or PMEL, however, may not be used
*  in any advertising or publicity to endorse or promote any products
*  or commercial entity unless specific written permission is obtained
*  from NOAA/PMEL/TMAP. The user also understands that NOAA/PMEL/TMAP
*  is not obligated to provide the user with any support, consulting,
*  training or assistance of any kind with regard to the use, operation
*  and performance of this software nor to provide the user with any
*  updates, revisions, new versions or "bug fixes".
*
*  THIS SOFTWARE IS PROVIDED BY NOAA/PMEL/TMAP "AS IS" AND ANY EXPRESS
*  OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED
*  WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
*  ARE DISCLAIMED. IN NO EVENT SHALL NOAA/PMEL/TMAP BE LIABLE FOR ANY SPECIAL,
*  INDIRECT OR CONSEQUENTIAL DAMAGES OR ANY DAMAGES WHATSOEVER
*  RESULTING FROM LOSS OF USE, DATA OR PROFITS, WHETHER IN AN ACTION OF
*  CONTRACT, NEGLIGENCE OR OTHER TORTUOUS ACTION, ARISING OUT OF OR IN
*  CONNECTION WITH THE ACCESS, USE OR PERFORMANCE OF THIS SOFTWARE. 
*
*
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
*       2/99 changed UNRAVEL to XUNRAVEL. Added YUNRAVEL, ZUNRAVEL, TUNRAVEL
* V500: 7/99 *sh* - use of "grid_line(1,mgrid_abstract)" changed to
*               "grid_line(1,mgrid_xabstract)" since mgrid_abstract changes
*               with SET GRID
* V530: 10/00 *sh* - bug fix: need to pass an *array* of argument contexts
*       to the C routine efcn_get_custom_axes
*                  - changed EFCN_GET_CUSTOM_AXES so it returns a status
* V541: 2/02 *sh* - support for modulo length
*       *acm* 3/12 Add E and F dimensions (use nferdims in tmap_dims.parm)
* 11/2022 *acm* Code cleanup: Modernize DO-loop syntax. 

        IMPLICIT NONE
	include 'tmap_dims.parm'
	include 'xtm_grid.cmn_text'
	include 'ferret.parm'
	include 'errmsg.parm'
	include 'grid_chg_fcns.parm'
	include 'EF_Util.parm'
	include 'xgrid_chg_fcns.cmn'
	include 'xtext_info.cmn'	! w/ ww_dim_name
	include 'xfr_grid.cmn'       ! w/ mgrid_abstract
	include 'xcontext.cmn'       ! for TIMES2
	include 'xvariables.cmn'     ! for TIMES2

* calling argument declarations
	INTEGER	gcfcn, cx_base, axes(*), status

* internal variable declarations
	LOGICAL known_limits
	INTEGER	CGRID_AXIS, TM_LENSTR1, CX_DIM_LEN, CAXIS_LEN,
     .          EFCN_GET_NUM_REQD_ARGS,
     .		cx, i, idim, axis_will_be(nferdims), n, grid, modulo
	REAL*8	TM_WORLD, dlo, dhi, del
	CHARACTER*40 GCF_NAME, buff, unit
        INTEGER EF_cx_list(EF_MAX_ARGS), narg

* Upon entry to this routine the context stack should be organized as
*	 	  default context (for this uvar)
*	 big_cx:  the merged grid context
* -->	cx_base:  argument N context
*		  argument N-1 context
*		  ...
*		  argument 1 context

* initialize
* ... consolidate info about function into local array to blur distinctions
*     between internal and external functions
	IF ( gcfcn .LE. gfcn_num_internal ) THEN
	   DO idim = 1, nferdims
	      axis_will_be(idim) = gfcn_axis_will_be(idim,gcfcn)
 	   ENDDO
	ELSE
	   call efcn_get_axis_will_be(gcfcn, axis_will_be)
	ENDIF

* process what axes we can: either "merged" (determined by grid stack logic),
*	NORMAL, and ABSTRACT ... but not IMPOSED ("supplied herein")
	DO idim = 1, nferdims

	   IF (axis_will_be(idim) .EQ. pgc_axis_is_abstract) THEN
	      axes(idim) = grid_line(1,mgrid_xabstract)         ! 7/99 mod
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

	ENDDO

* *** XUNRAVEL, YUNRAVEL, ZUNRAVEL, TUNRAVEL, EUNRAVEL, FUNRAVEL functions ***
* The result of these functions are grids of ABSTRACT and NORMAL axes
*   .... 7/97 changes -- handled entirely within "typical" cases above

	IF ( gcfcn .EQ. p_reshape ) THEN
* *** RESHAPE function ***
* The second argument contains the desired grid. (It cannot simply be merged,
* however, because the result limits have no obvious relation to the inputs)
	   cx = cx_base 	! component 2: the reshaping grid
	   DO idim = 1, nferdims
	      axes(idim) = CGRID_AXIS ( idim, cx )
	   ENDDO

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
     .                                  'Hertz', .FALSE., 0.0D0,
     .                                  axes(t_dim), status)
	   IF (status .NE. ferr_ok) GOTO 5000


**************
*     FOR EXTERNAL FUNCTIONS


	ELSEIF (gcfcn .GT. gfcn_num_internal ) THEN

*   ... create a context list array
           narg = EFCN_GET_NUM_REQD_ARGS(gcfcn)
           DO i = 1, narg
              EF_cx_list(i) = cx_stack_ptr - i + 1
	   ENDDO

           IF (axis_will_be(1) .EQ. pgc_axis_supplied_herein
     .          .OR. axis_will_be(2) .EQ. pgc_axis_supplied_herein
     .          .OR. axis_will_be(3) .EQ. pgc_axis_supplied_herein
     .          .OR. axis_will_be(4) .EQ. pgc_axis_supplied_herein
     .          .OR. axis_will_be(5) .EQ. pgc_axis_supplied_herein
     .          .OR. axis_will_be(6) .EQ. pgc_axis_supplied_herein) THEN
              CALL EFCN_GET_CUSTOM_AXES(gcfcn, EF_cx_list, status)
              IF (status .NE. ferr_ok) CALL ERRMSG
     .                          ( status, status, ' ', *5000) 
           ENDIF

           DO idim = 1, nferdims

              IF (axis_will_be(idim) .EQ. pgc_axis_supplied_herein) THEN
                 CALL EFCN_GET_CUSTOM_AXIS(gcfcn, idim, dlo, dhi, del,
     .                unit, modulo)
                 CALL TM_MAKE_DYN_LINE(idim, dlo, dhi, del,
     .                unit, modulo, 0.0D0, axes(idim), status)
                 IF (status .NE. ferr_ok) GOTO 5000
              ENDIF

	   ENDDO

	ENDIF

* successful completion
	status = ferr_ok
	RETURN

* error exit
 5000	RETURN
	END
