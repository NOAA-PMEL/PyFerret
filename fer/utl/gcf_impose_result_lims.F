	SUBROUTINE GCF_IMPOSE_RESULT_LIMS( gcfcn, memory, mr_list,
     .          cx_list, res_cx, status )

* Complete the context for the result of a grid-changing function.
* Fill in the limits for those axes of the grid changing function which
* were "imposed" by the function.
* Reconcile the limits of the arguments with the limits of the result where
* offsets are imposed.

* the addition (or major modification) of an internal grid-changing function
* requires changes to these routines:
*	grid_chg_fcns.parm	- parameters to number the functions
*	XGRID_CHG_FCNS_DATA	- description of the function and args
*	DO_INTERNAL_GC_FCN	- code to execute the function
* (opt)	GCF_GET_IMPOSED_AXES	- return the "imposed" axes, if any
* (opt)	GCF_IMPOSE_RESULT_LIMS	- fill in limits of "imposed" axes, if any

* V450: 1/97
* *jc* 12/97 adding external function capabilities
* V491: 1/98 - added TIMES2, which imposes its T axis
* *jc* 01/98 put includes first

* debug declarations for TIMES2
	INTEGER	n_com1, n_grid, com1_grid
	REAL*8	TM_WORLD, dlo_com1, dlo_grid, dhi_com1, dhi_grid

#ifdef unix
	include 'ferret_cmn/ferret.parm'
	include 'ferret_cmn/errmsg.parm'
	include 'ferret_cmn/grid_chg_fcns.parm'
	include 'ferret_cmn/xcontext.cmn'
	include 'ferret_cmn/xvariables.cmn'
	include 'ferret_cmn/xgrid_chg_fcns.cmn'
#else
	INCLUDE 'FERRET_CMN:FERRET.PARM'
	INCLUDE 'FERRET_CMN:ERRMSG.PARM'
	INCLUDE 'FERRET_CMN:GRID_CHG_FCNS.PARM'
	INCLUDE 'FERRET_CMN:XCONTEXT.CMN'
	INCLUDE 'FERRET_CMN:XVARIABLES.CMN'
	INCLUDE 'FERRET_CMN:XGRID_CHG_FCNS.CMN'
#endif

* calling argument declarations
	INTEGER	gcfcn, cx_list(*), mr_list(*), res_cx, status
	REAL	memory( mem_blk_size, max_mem_blks )

* internal variable declarations
	LOGICAL fix_me(4)
	INTEGER	CGRID_SIZE, CAXIS_LEN, CGRID_AXIS, CX_DIM_LEN,
     .		com1_cx, com2_cx, idim, filled, size, uvar,
     .		res_grd, outlen(4), accum(2:4),
     .		lo_off(4),  hi_off(4), action(4),
     .		axis_will_be(4), lo_lim, hi_lim

* initialize
	uvar = cx_variable(res_cx)
	res_grd = cx_grid(res_cx)


* Fill in the limits for those axes of the grid changing function which
* were "imposed" by the function.

* (1/98)  *** all CUSTOM axes ***
* If the function creates and imposes a custom axis (see GCF_GET_IMPOSED_AXES)
*     then result lims are assumed BY DEFAULT to fill the limits of that axis
* The function itself may over-ride this assumption and impose other limits
*     or detect an error.
	DO 100 idim = 1, 4
	   IF (gcfcn .LE. gfcn_num_internal ) THEN
* ... internal function imposing custom axis?
	     IF (  gfcn_axis_will_be(idim,gcfcn)
     .	      .NE. pgc_axis_supplied_herein) GOTO 100	! 100 if not custom
	   ELSE
* ... external function imposing custom axis?
*             STOP 'EF code must GOTO 100 if not custom axis'
	      GOTO 100
	   ENDIF

* get size of custom output axis
	   size = CAXIS_LEN ( idim, res_cx )

* impose full length on result
	   cx_lo_ss(res_cx, idim) = 1
	   cx_hi_ss(res_cx, idim) = size
	   cx_by_ss(idim, res_cx) = .TRUE.
	   CALL FLESH_OUT_AXIS( idim, res_cx, status )
	   IF ( status .NE. ferr_ok ) GOTO 5000
 100	CONTINUE

* *** UNRAVEL function ***
	IF ( gcfcn .EQ. p_unravel ) THEN
* X axis is imposed as ABSTRACT
* size of output on X is total size of input of component
	   com1_cx = cx_list(1) 	! only one argument to function
	   cx_lo_ss(res_cx,x_dim) = 1
	   cx_hi_ss(res_cx,x_dim) = CGRID_SIZE(com1_cx)
	   cx_by_ss(x_dim, res_cx) = .TRUE.
	   CALL FLESH_OUT_AXIS( x_dim, res_cx, status )
	   IF ( status .NE. ferr_ok ) GOTO 5000
	

* *** TIMES2 (debug) function ***
	ELSEIF ( gcfcn .EQ. p_times2 ) THEN
* T axis gets custom-inserted
	   com1_cx = cx_list(1) 	! only one argument to function
	   com1_grid = cx_grid(com1_cx)

* check that result axis limits match the argument
* ... argument range
	   n_com1   = CX_DIM_LEN(t_dim, com1_cx)
	   dlo_com1 = TM_WORLD(cx_lo_ss(com1_cx,t_dim), com1_grid, t_dim,
     .						box_middle )
	   dhi_com1 = TM_WORLD(cx_hi_ss(com1_cx,t_dim), com1_grid, t_dim,
     .						box_middle )

* ... custom axis range
	   n_grid   = CAXIS_LEN(t_dim, res_cx)
	   dlo_grid = TM_WORLD(1,      res_grd, t_dim, box_middle )
	   dhi_grid = TM_WORLD(n_grid, res_grd, t_dim, box_middle )

* sanity checking -- messages to help with debugging
	   IF (n_com1   .NE. n_grid  )
     .			CALL WARN('TIMES2 grid size mismatch')
	   IF (dlo_com1 .NE. dlo_grid)
     .			CALL WARN('TIMES2 low limit mismatch')
	   IF (dhi_com1 .NE. dhi_grid)
     .			CALL WARN('TIMES2 high limit mismatch')



* *** RESHAPE function ***
* Determine how far the input data will fill the result grid
* Note that for those axes where the result axis is the same as the comp 1 axis
* the arg 1 limits should be preserved
* e.g. RESHAPE(SST,G_Z12_TYEAR) should preserve the X and Y limits
	ELSEIF ( gcfcn .EQ. p_reshape ) THEN
	   com1_cx = cx_list(1) 	! the input data
	   size = CGRID_SIZE(com1_cx)

* ... get "action" which tells which result axes inherit the comp 1 limits
	   CALL GCF_GET_ARG_LIMS (gcfcn, 1, com1_cx, res_grd,
     .		lo_off,  hi_off, action, status)
	   IF ( status .NE. ferr_ok) GOTO 5000

* ... determine the size of the output "container"
	   com2_cx = cx_list(2) 	! the reshaping grid
	   DO 310 idim = 1, 4
	      IF ( action(idim) .EQ. ppass_along ) THEN
	         outlen(idim) = CX_DIM_LEN(idim, com2_cx)
	      ELSE
	         outlen(idim) = CAXIS_LEN (idim, com2_cx)
	      ENDIF
 310	   CONTINUE

* set default limits: full ax len or match com1 depending on "action"
	   DO 320 idim = 1, 4
	      fix_me(idim) = CGRID_AXIS(idim, res_cx) .NE. mnormal
	      IF ( fix_me(idim) ) THEN
	        IF ( action(idim) .EQ. ppass_along ) THEN
	           cx_lo_ss(res_cx,idim) = cx_lo_ss(com1_cx,idim)
	           cx_hi_ss(res_cx,idim) = cx_hi_ss(com1_cx,idim)
	        ELSE
	           cx_lo_ss(res_cx,idim) = 1
	           cx_hi_ss(res_cx,idim) = outlen(idim)
	        ENDIF
	      ELSE
	         cx_lo_ss(res_cx,idim) = unspecified_int4
	         cx_hi_ss(res_cx,idim) = unspecified_int4
	      ENDIF
* ... all axes of length 1 provide no opportunity for adjusting output
	      IF (outlen(idim) .EQ. 1) fix_me(idim) = .FALSE.
 320	   CONTINUE

* accumulate the axis lengths to help with calculation
* Note: this can cause integer overflow for grid with ABSTRACT axes
*	a checp fix (7/97) checks for size above 9999999
	   accum(2) = outlen(1)
	   DO 330 idim = 3, 4
	      IF (accum(idim-1) .GE. 9999999
     .	     .OR. outlen(idim-1) .GE. 9999999 ) THEN
	         accum(idim) = 9999999
	      ELSE
	         accum(idim) = accum(idim-1) * outlen(idim-1)
	      ENDIF
 330	   CONTINUE
* figure out how high the input data really fills the output grid
* Note that the input data may not fit into an integer number of output
* rows and columns. If not, round up to allow room, and fill bad flags in
* DO_INTERNAL_GC_FCN, to complete the unfinished rows.
	   DO 340 idim = 4, 2, -1
	      filled = size/accum(idim)   ! integer arithmatic
	      IF ( filled.GT.0 .AND. fix_me(idim) ) THEN
	         IF ( size .GT. filled*accum(idim) ) filled = filled + 1
	         cx_hi_ss(res_cx,idim) = MIN(outlen(idim), filled) +
     .					 cx_lo_ss(res_cx,idim) - 1
	         GOTO 360	! break from loop
	      ELSE
	         cx_hi_ss(res_cx,idim) = cx_lo_ss(res_cx,idim)
	      ENDIF
 340	   CONTINUE
* ... it all fits into the lowest order row (which may not be the X axis)
	   DO 350 idim = 1, 4
	      IF (fix_me(idim)) THEN
	         cx_hi_ss(res_cx,idim) = cx_lo_ss(res_cx,idim) - 1
     .				       + size
	         GOTO 360
	      ENDIF
 350	   CONTINUE

* flesh out the axes
 360	   DO 370 idim = 1, 4
	      cx_by_ss(idim, res_cx) = .TRUE.
	      CALL FLESH_OUT_AXIS( idim, res_cx, status )
	      IF ( status .NE. ferr_ok ) GOTO 5000
 370	   CONTINUE

*
* THIS NEXT SECTION DEALS WITH EXTERNAL FUNCTIONS
*

	ELSEIF ( gcfcn .GT. gfcn_num_internal ) THEN

* ... does the external function need to impose special size limits
*     onto the result? (perhaps a parameter of the function determines this?)
* ... If so, impose 'em here

* What kind of axes are we dealing with?
	   CALL EFCN_GET_AXIS_WILL_BE(gcfcn, axis_will_be)

	   DO 450 idim = 1, 4


* If the axis is abstract ... 
*   get the lo and hi limits for this axis
*   create one as in UNRAVEL above

	      IF (axis_will_be(idim) .EQ. pgc_axis_is_abstract)	THEN

		 CALL EFCN_GET_AXIS_ABSTRACT(gcfcn, memory, mr_list, 
     .           cx_list, idim, lo_lim,hi_lim)
		 cx_lo_ss(res_cx,idim) = lo_lim
		 cx_hi_ss(res_cx,idim) = hi_lim
		 cx_by_ss(idim, res_cx) = .TRUE.
		 CALL FLESH_OUT_AXIS( idim, res_cx, status )
		 IF ( status .NE. ferr_ok ) GOTO 5000


* If the axis is normal ... nothing needs to be done

*	      ELSEIF (axis_will_be(idim) .EQ. pgc_axis_is_normal) THEN
*       do nothing


* If the axis is custom ... not supported yet

	      ELSEIF (axis_will_be(idim) .EQ. pgc_axis_supplied_herein)
     .                THEN
		 CALL ERRMSG( ferr_internal, status,
     .                'gcf_impose_result_lims: external function'// 
     .                'custom axes not supported', *5000 )


* If the axis is implied_by_args ... nothing needs to be done

*	      ELSEIF (axis_will_be(idim) .EQ. pgc_axis_implied_by_args)
*     .             THEN
*       do nothing

	      ENDIF

 450	   CONTINUE

	ELSE

	   CONTINUE	! this function does not impose any limits

	ENDIF

c* *jc* 01/98 Removing this section to allow for arguments which are constants.
c*
	RETURN

********************************
* Reconcile the limits of the arguments with the limits of the result where
* offsets are imposed.
* At this point there is one valid reason for an incomplete result context --
* that is, that the function requests "offset" limits from its arguments.
* Offsets are used, for example, in taking a centered derivative -- an offset
* of an additional point is needed in the argument relative to the result.
* However, the process of applying the offsets introduces ambiguities -- e.g.
* with an offset of hi_lim+1, a request for the full axis range of the
* function result and a request for the full range minus one will both lead
* to a full range argument.
* To sidestep the ambiguities axes of arguments with offset limits are not
* merged in MERGE_CONTEXT -- instead they are merged here when all of the
* argument limits are available at the same moment.

* is the result context incomplete?

JC	DO 1000 idim = 1, 4
JC	   IF ( cx_lo_ww(idim,res_cx) .EQ. unspecified_val8 ) THEN
* only UVARs end up in this routine so uvar_given is available
JC	      IF (uvar_given(idim,uvar).EQ.uvlim_irrelevant) GOTO 1000

* ... logic for offsets not yet implemented -- take internal error
* some ideas:
*  i) normal behavior is to impose argument minus offset as limits
* ii) ambiguities will be permitted only within offset points of axis edge
*     else they are errors.
*iii) if ambiguous at edges how to resolve??? Need to know the **requested**
*     limits

JC	      CALL ERRMSG( ferr_internal, status,
JC     .			'gcf_get_result_lims: incomplete cx', *5000 )

JC	   ENDIF
JC 1000	CONTINUE

JC	RETURN


* error exit
 5000	RETURN
	END
