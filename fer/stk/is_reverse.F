	SUBROUTINE IS_REVERSE( memory, *, *, status )

* reverse the ordering of the data along the axes indicated in ds_ordering
* and delete the (backwards) component data from memory

* programmer - steve hankin
* NOAA/PMEL, Seattle, WA - Tropical Modeling and Analysis Program
*
* V320: 2/16/95
* V411: 9/11/95 (IBM port) array "reversed" is declared wrong type

#ifdef unix
	include	'ferret_cmn/ferret.parm'
	include	'ferret_cmn/errmsg.parm'
	include	'ferret_cmn/interp_stack.parm'
	include	'ferret_cmn/xcontext.cmn'
	include	'ferret_cmn/xvariables.cmn'
	include	'ferret_cmn/xprog_state.cmn'
#else
	INCLUDE	'FERRET_CMN:FERRET.PARM'
	INCLUDE	'FERRET_CMN:ERRMSG.PARM'
	INCLUDE	'FERRET_CMN:INTERP_STACK.PARM'
	INCLUDE	'FERRET_CMN:XCONTEXT.CMN'
	INCLUDE	'FERRET_CMN:XVARIABLES.CMN'
	INCLUDE	'FERRET_CMN:XPROG_STATE.CMN'
#endif

* calling argument declarations:
	INTEGER	status
	REAL	memory( mem_blk_size, max_mem_blks )
* normal return is for error conditions
* exit #1 is to request another grid ( stack has been pushed )
* exit #2 is to signal completion of the activity at this stack level

*   ... interpretation stack setup at the time this is called ...
*	stack level		stack contents
*	isp+1	<--	level of last activity ( component just computed )
*	 isp	<--	current stack level (for moduloing )

* internal variable declarations:
	LOGICAL	short_span
	INTEGER CAXIS_LEN, TM_AXES_REVERSED, nrev,
     .		com_mr, res_mr, com_cx, res_cx,	com_isp, res_isp,
     .		axis, i, idim, var, hi_ss, lo_ss, itmp, reversed(4)

* --- end of introductory code ---

* Routine IS_STRIP requests the reversed component data under
* the special category cx_category=cat_revfile

* initialize
	res_isp    = isp
	com_isp    = isp + 1
	com_mr     = is_mr( com_isp )
	res_cx     = is_cx( res_isp )
	com_cx     = is_cx( com_isp )
	axis       = is_axis( res_isp )
	var	   = cx_variable(com_cx)

* get the list of reversed axes
	nrev = TM_AXES_REVERSED(var, reversed)

* diagnostic mode output: " doing modulo VAR_NAME[x=lo:hi,D=#]"
	IF ( mode_diagnostic ) 	CALL DIAG_OP
     .				( 'doing',isact_class_reverse, res_cx, axis )

* create the space for the result in memory
	CALL CREATE_MEM_VAR( res_cx, res_mr, status )
	IF ( status .NE. ferr_ok ) GOTO 5100
	is_mr( res_isp ) = res_mr

* the component data thinks that its subscripts are from "the other end" of
* the axis (from whence the data were read).  In order to prevent subscript
* errors we will disabuse the data of that misconception atthis point
	DO 100 i = 1, nrev
	   axis = reversed(i)
	   CALL VAR_SS_LIMS( axis, res_cx, lo_ss, hi_ss )
	   itmp = mr_lo_ss( com_mr, axis )
	   mr_lo_ss(com_mr,axis) = hi_ss + lo_ss - mr_hi_ss(com_mr,axis)
	   mr_hi_ss(com_mr,axis) = hi_ss + lo_ss - itmp
 100	CONTINUE

* copy the source data into the result
	CALL COPY_GRID(
     .    memory( 1, mr_blk1(com_mr) ), com_mr,
     .    memory( 1, mr_blk1(res_mr) ), res_mr   )

* reverse the result along all of the required axes
	DO 200 i = 1, nrev
	   CALL REVERSE_GRID_AXIS
     .		( reversed(i), res_cx, res_mr,memory(1, mr_blk1(res_mr)) )
 200	CONTINUE

* successful completion of modulo operation
 1000	CALL MR_NOT_IN_USE( com_mr )	! remove memory protection on component
	CALL DELETE_VARIABLE( com_mr )  ! delete reversed component data
	cx_stack_ptr = cx_stack_ptr - 1 ! pop com_cx from context stack
	status = ferr_ok
	RETURN 2

* error exits
 5000	RETURN
 5100	CALL MR_NOT_IN_USE( com_mr )
	CALL DELETE_VARIABLE( com_mr )  ! delete reversed component data
	GOTO 5000
	END	

