	SUBROUTINE GCF_IMPOSE_RESULT_LIMS( gcfcn, cx_list, res_cx, status )

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

* calling argument declarations
	INTEGER	gcfcn, cx_list(*), res_cx, status

* internal variable declarations
	LOGICAL fix_me(4)
	INTEGER	CGRID_SIZE, CAXIS_LEN, CGRID_AXIS, CX_DIM_LEN,
     .		com1_cx, com2_cx, idim, filled, size, uvar,
     .		res_grd, outlen(4), accum(2:4),
     .		lo_off(4),  hi_off(4), action(4)

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

* initialize
	uvar = cx_variable(res_cx)
	res_grd = cx_grid(res_cx)

* Fill in the limits for those axes of the grid changing function which
* were "imposed" by the function.

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

	ELSEIF ( gcfcn .GT. gfcn_num_internal ) THEN

* ... does the external function need to impose special size limits
*     onto the result? (perhaps a parameter of the function determines this?)
* ... If so, impose 'em here
	   STOP 'gcf_impose_result_lims: external functions not implemented'

	ELSE

	   CONTINUE	! this function does not impose any limits

	ENDIF

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
	DO 1000 idim = 1, 4
	   IF ( cx_lo_ww(idim,res_cx) .EQ. unspecified_val8 ) THEN
* only UVARs end up in this routine so uvar_given is available
	      IF (uvar_given(idim,uvar).EQ.uvlim_irrelevant) GOTO 1000

* ... logic for offsets not yet implemented -- take internal error
* some ideas:
*  i) normal behavior is to impose argument minus offset as limits
* ii) ambiguities will be permitted only within offset points of axis edge
*     else they are errors.
*iii) if ambiguous at edges how to resolve??? Need to know the **requested**
*     limits

	      CALL ERRMSG( ferr_internal, status,
     .			'gcf_get_result_lims: incomplete cx', *5000 )

	   ENDIF
 1000	CONTINUE

	RETURN

* error exit
 5000	RETURN
	END
