	SUBROUTINE IS_DO_GC_FCN
     .			( memory, alg_type, alg_text, lo_isp, status )

* do bookkeeping and call up calculation to perform grid changing fcn
* (which may be an externally-defined function)

* V450:  1/97 - based on IS_DO_OP
*        7/97 - added status and memory args to DO_INTERNAL_GC_FCN call
*       7/25/97 - increase number of args allowed from 5 to 10

#ifdef unix
	include	'ferret_cmn/ferret.parm'
	include	'ferret_cmn/errmsg.parm'
	include	'ferret_cmn/rpn.parm'
	include	'ferret_cmn/xvariables.cmn'
	include	'ferret_cmn/xcontext.cmn'
	include	'ferret_cmn/xalgebra.cmn'
	include	'ferret_cmn/xtext_info.cmn'
	include	'ferret_cmn/xfr_grid.cmn'	! w/ mgrid_abstract
#else
	INCLUDE	'FERRET_CMN:FERRET.PARM'
	INCLUDE	'FERRET_CMN:ERRMSG.PARM'
	INCLUDE	'FERRET_CMN:RPN.PARM'
	INCLUDE	'FERRET_CMN:XVARIABLES.CMN'
	INCLUDE	'FERRET_CMN:XCONTEXT.CMN'
	INCLUDE	'FERRET_CMN:XALGEBRA.CMN'
	INCLUDE	'FERRET_CMN:XTEXT_INFO.CMN'
	INCLUDE	'FERRET_CMN:XFR_GRID.CMN'	! W/ MGRID_ABSTRACT
#endif

* calling arguments
	INTEGER		alg_type, lo_isp, status
	REAL		memory( mem_blk_size, max_mem_blks )
	CHARACTER*(*)	alg_text

* local variables
	LOGICAL	ITS_UN_CACHED, not_reusable
	INTEGER	CX_DIM_LEN, STR_MATCH, GCF_FIND_FCN, GCF_NUM_ARGS,
     .		CGRID_AXIS,
     .		ifcn, iop, ilogic, icom, idim, num_com, mr, res_cx,
     .		mr1, mr2, mr3, mr4, mr5, mr6, mr7, mr8, mr9, impl_grid,
     .		frame_bottom, res_len, com_len, com_cx, res_mr

* convenience equivalences
	INTEGER	com(10)
	EQUIVALENCE (com(1),mr1), (com(2),mr2), (com(3),mr3), (com(4),mr4),
     .              (com(5),mr5),
     .		    (com(6),mr6), (com(7),mr7), (com(8),mr8), (com(9),mr9),
     .              (com(10),res_mr)

*        ... interpretation stack setup at the time this is called ...
*	stack level		stack contents
*	 isp	<--	most recently obtained component for calculation
*	isp-1		...
*	isp-2	<--	...	other components of this expression on stack
*	isp-n		...
*	lo_isp	<--	level at which expression evaluation was called
*
* eg. for tau ^ 2 as a part of the full expression
*			level isp     will contain the constant 2
*			level isp-1   will contain tau
*		and the alg_type/alg_text will point to "^"

* the "frame_bottom" is the isp level that will be left when the components of
* the current calculation (1 to 3) are reduced to a single value.
* on exit isp will point to the frame bottom and the result will be left there

* identify the text of the particular function
*	(internal or external GC fcn handled the same)
	ifcn = GCF_FIND_FCN( alg_text )
	IF ( ifcn .EQ. unspecified_int4 ) STOP ' bad GC fcn err'
	num_com = GCF_NUM_ARGS( ifcn )

* sanity check on stack contents
	frame_bottom = isp - num_com + 1
	IF ( frame_bottom .LE. lo_isp ) CALL ERRMSG
     .		( ferr_syntax, status, 'RPN', *5000 )

!* the merged context is already nearly complete
	res_cx = is_big_cx(lo_isp)

* need to fill in those axes that the function "imposes"
	CALL GCF_IMPOSE_RESULT_LIMS( ifcn,
     .					is_cx(lo_isp+1), res_cx, status )
	IF ( status .NE. ferr_ok ) GOTO 5000

* allocate memory space for result
	CALL CREATE_MEM_VAR( res_cx, res_mr, status )
	IF ( status .NE. ferr_ok ) GOTO 5000
	mr_category( res_mr ) = cat_temp_var
	mr_variable( res_mr ) = res_mr	! cat_temp_var vars point to themselves

* point to the memory resident components
* (if fewer than 9 components default extras to dummy)
	DO 300 icom = 1, num_com
 300	com(icom) = is_mr(frame_bottom+icom-1)
	DO 310 icom = num_com+1, 9
 310	com(icom) = dummy_mr

* set up 4D array limits as unsubscripted variables in COMMON
	CALL NON_ARRAY_SUBSC( com, 10 )

* perform the operation
	CALL DO_INTERNAL_GC_FCN( memory,
     .			ifcn, num_com, is_cx(frame_bottom),
     .			memory(1, mr_blk1(mr1) ),   mr1,
     .			memory(1, mr_blk1(mr2) ),   mr2,
     .			memory(1, mr_blk1(mr3) ),   mr3,
     .			memory(1, mr_blk1(mr4) ),   mr4,
     .			memory(1, mr_blk1(mr5) ),   mr5,
     .			memory(1, mr_blk1(mr6) ),   mr6,
     .			memory(1, mr_blk1(mr7) ),   mr7,
     .			memory(1, mr_blk1(mr8) ),   mr8,
     .			memory(1, mr_blk1(mr9) ),   mr9,
     .			memory(1, mr_blk1(res_mr)), res_mr,
     .			status )
	IF ( status .NE. ferr_ok ) GOTO 5100

* clean up 

* ... if any components were un_cached we must uncache the result
	not_reusable = .FALSE.
	DO 2050 icom = isp, frame_bottom, -1
 2050	IF ( ITS_UN_CACHED(is_mr(icom)) ) not_reusable = .TRUE.
	IF ( not_reusable ) CALL UN_CACHE( res_mr )

* ...release component cx stack space
	DO 2100 isp = isp, frame_bottom, -1
	   IF ( is_cx(isp) .NE. cx_stack_ptr ) WRITE (6,*) 'OP_STK??'	! temp
	   cx_stack_ptr = cx_stack_ptr - 1
	   mr  = is_mr( isp )
	   IF ( mr_protected(mr) .EQ. mr_temporary
     .	   .OR. mr_category (mr) .EQ. cat_temp_var ) THEN
	      CALL DELETE_VARIABLE( mr )
	   ELSE
	      CALL MR_NOT_IN_USE( mr )
	   ENDIF
* (9/95) deallocate dynamic grids from components
	   com_cx = is_cx(isp)
	   IF ( cx_has_impl_grid(com_cx) ) THEN
	      impl_grid = cx_grid(com_cx)
	      CALL DEALLO_DYN_GRID(impl_grid)
	      cx_has_impl_grid(com_cx) = .FALSE.
	   ENDIF
 2100	CONTINUE

* collapse the interp stack frame
	isp = frame_bottom
	is_mr( isp ) = res_mr

* save the result context at the cx level previously used by the 1st component
	cx_stack_ptr = cx_stack_ptr + 1
	CALL TRANSFER_CONTEXT( res_cx, cx_stack_ptr )
	is_cx( isp ) = cx_stack_ptr

	RETURN

* error exit
 5000	RETURN
 5100	CALL DELETE_VARIABLE(res_mr)
	GOTO 5000
	END

