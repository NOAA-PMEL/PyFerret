	SUBROUTINE IS_DO_OP(  alg_type, alg_text, lo_isp, 
     ,                       ifv, status )

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
* do bookkeeping and call up calculation to perform op_num between
* 0,1,2 or 3 grids

* programmer - steve hankin
* NOAA/PMEL, Seattle, WA - Tropical Modeling and Analysis Program
*
* revision 0.0 - 2/13/87 - based on ALG_DO_OP rev 1.0
* revision 1.0 - 8/11/87 - allowed for expanding grids: lines to planes, etc.
* revision 1.1 - 9/30/87 - added mr_perm_prot
* revision 1.2 - 2/18/88 - bug fix: LOAD/i=lo:hi/j=j1/k=k1/l=lo:hi temp,salt
*				    LIST/i=i1 temp salt +
*				  --> dimension error: tries to expand temp,salt
*				and check the data planes before GRID_OPERATIO*
* revision 1.3 - 7/8/88  - make sure subscript limits are adequate
* revision 1.4 - 1/31/89 - passed num_com to GRID_OPERATION
* V200:  6/20/89 - re-worked user defined variable logic
*	10/16/89 - set up adjustable array limits with NON_ARRAY_SUBSC
*	11/27/89 - IF-THEN-ELSE logic
* V230:   8/7/92 - upped to 4 the number of arguments a function can use
* V312: 5/94 - array "memory" as a calling argument
* V400: corrected bad type declaration of STR_MATCH (found by IBM compiler)
* V420: 9/95 - deallocate dynamic grids from components that had 'em
*       10/5 ... a subtle change required by this is that the result must not
*	       explicitly require the same grid as the first component  (lest
*	       the result get deleted when that grid is deallocated.  In
*	       fact, the true result grid and context are created by the
*	       individual mergers of each component ... never by examining
*	       the grid that results from this routine.  Thus (experimentally)
*	       we will always represent the output of this routine as being
*	       on grid ABSTRACT.  Hopefully, there will be no side-effects.
*	11/95 - if any of the components are non-reusabe ("un_cached") so
*		must be the result
* V500:  1/99 *sh* Trap as errors attempts to use string variables as
*		arguments. Strings may be passed only to grid-changing
*		functions (see IS_DO_GC_FC)
* V530:  8/00 *sh* added support for string arrays
* V533:  7/01 *sh* improved support for string arrays -- logical operations
* V630:  9/09 *acm* Fix bug 1380: If the operation is a EQ,NE,LT,LE,GT,GE on 
*                 string arguments, the result is numeric. Set cx_type(res_cx) 
*                 accordingly, then compound conditinals will work.
* V671 7/11  *acm* Ticket 1868 IFV - masking expressions with ifValid, treating 
*		   zero as valid.
*       *acm* 3/12 6D Ferret (common uses nferdims in tmap_dims.parm)
* V68  3/12  *acm* The string operations EQ, NE, LT, etc return a float-type result.
*                  Reset mr_type; it has already been set to string elsewhere.
* V694 5/14  *acm* Ticket 2234: Issue an error instead of returning missing results on 
*                  inappropriate use of string arguments in numeric expressions or mixed
*                  string/numeric expressions with logic operators. See bn_strings.jnl
* V720 5/17 *sh*  added support for operations on DSG ragged arrays 
* V76  1/20 *acm* Implementing Point-type DSG datasets 
* 11/2022 *acm* Code cleanup: Modernize DO-loop syntax. 

        IMPLICIT NONE
	include 'tmap_dims.parm'
	include	'xdset_info.cmn_text'
	include	'ferret.parm'
	include	'errmsg.parm'
	include	'rpn.parm'
	include	'xvariables.cmn'
	include	'xcontext.cmn'
	include	'xalgebra.cmn'
	include	'xtext_info.cmn'
	include	'xfr_grid.cmn'	! w/ mgrid_abstract

* calling arguments
	LOGICAL		ifv
	INTEGER		alg_type, lo_isp, status
	CHARACTER*(*)	alg_text

* local variables
	LOGICAL	ITS_UN_CACHED, TM_ITSA_DSG,
     .		not_reusable, arg_types_matched, numerical_op, its_dsg
	INTEGER	CX_DIM_LEN, STR_MATCH, MGRID_SIZE_DELTA,
     .		TM_DSG_DSET_FROM_GRID, TM_DSG_NFEATURES,
     .		ifcn, iop, ilogic, icom, idim, num_com, mr, res_cx,
     .		mr1, mr2, mr3, mr4, impl_grid, arg_type, i1,
     .		frame_bottom, op_num, res_len, com_len, com_cx, res_mr,
     .		dsg_grid, dset, i8_size, nfeatures, nobs, orientation

* convenience equivalences
	INTEGER	com(5)
	EQUIVALENCE (com(1),mr1), (com(2),mr2), (com(3),mr3), (com(4),mr4),
     .              (com(5),res_mr)

* local parameter declarations
	INTEGER	     p_log_op_first, p_log_op_last  ! see operate.F
	PARAMETER  ( p_log_op_first = 8,
     .		     p_log_op_last = 13 )

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

* initialize DSG info
* since an abstract grid is substituted below for the result, and we may be
* operating on entirely intermediate results, we cannot follow the usual route
* of learning the DSG-ness of the calculation from the grid.  Instead we
* look at the grid of the result uvar
! TBD: not sure if it matters whether we use is_big_cx or is_cx
	dsg_grid = cx_grid(is_big_cx(lo_isp))
	its_dsg = TM_ITSA_DSG(dsg_grid)
	IF (its_dsg) THEN
          dset      = TM_DSG_DSET_FROM_GRID( dsg_grid )
          nfeatures = TM_DSG_NFEATURES( dsg_grid )
          nobs        = dsg_obs_dimlen(dset)
	  orientation = dsg_orientation(dset)
	ENDIF

* identify the text of the particular item within the algebraic class
* ... function
	IF     ( alg_type .EQ. alg_function   ) THEN
	   ifcn = STR_MATCH( alg_text, alg_fcn, num_functions )
	   IF ( ifcn .EQ. atom_not_found ) STOP ' bad fcn err'
	   num_com = alg_num_fcn_args( ifcn )
	   op_num  = ifcn + offset_functions
* ... operator
	ELSEIF ( alg_type .EQ. alg_operator   ) THEN
	   iop = STR_MATCH ( alg_text, alg_op, num_operators )
	   IF ( iop .EQ. atom_not_found ) STOP ' bad op err'
	   num_com = 2
	   op_num  = iop
* ... logic structure
	ELSEIF ( alg_type .EQ. alg_log_struct .OR.
     .           alg_type .EQ. alg_log_vstruct ) THEN
	   ilogic = STR_MATCH (	alg_text, alg_struct, num_log_structs )
	   IF ( ilogic .EQ. atom_not_found ) STOP ' bad if-then err'
	   IF ( ilogic .EQ. pthen ) THEN
	      num_com = 2	! IF ... THEN ...
	   ELSEIF ( ilogic .EQ. pelse ) THEN
	      num_com = 3	! IF ... THEN ... ELSE ...
	   ELSE
	      STOP 'bad if'
	   ENDIF
	   op_num  = ilogic + offset_logic_struc
	   IF (alg_type .EQ. alg_log_vstruct) op_num = op_num + 2
	ELSE
	   STOP '??? fcn or op'
	ENDIF
	frame_bottom = isp - num_com + 1
	IF ( frame_bottom .LE. lo_isp ) CALL ERRMSG
     .		( ferr_syntax, status, 'RPN', *5000 )

* classify the data type of the arguments
* ... ignore type of "cond" in IF cond THEN result
	IF ( alg_type .EQ. alg_log_struct .OR.
     .       alg_type .EQ. alg_log_vstruct ) THEN
	  i1 = 2 ! ignore "cond"
	ELSE
	  i1 = 1
	ENDIF
	arg_type =  cx_type(is_cx(frame_bottom+i1-1))
	arg_types_matched = .TRUE.
	DO icom = i1, num_com
	   com_cx  = is_cx(frame_bottom+icom-1)
	   IF (arg_type .NE. cx_type(com_cx) ) THEN
	     arg_types_matched = .FALSE.
	     arg_type = ptype_float
	   ENDIF
	ENDDO

* allocate context space for the result
	CALL STACK_PTR_UP( cx_stack_ptr, max_context, status )
	IF ( status .NE. ferr_ok ) RETURN
	res_cx = cx_stack_ptr

* use first component context as the default
	CALL TRANSFER_CONTEXT( is_cx(frame_bottom), res_cx )
	cx_category( res_cx ) = cat_temp_var
	cx_variable( res_cx ) = 1		! in case it is a constant
	cx_grid    ( res_cx ) = mgrid_abstract	! see 10/5 notes above
	cx_type    ( res_cx ) = arg_type

* stretch sizes to maxes for all components on each axis
* e.g. temp[k=1] + temp[K=1:3] needs 3 in the K direction
* also - if there is a meaningful axis then use it
* e.g. I[i=1] + J[j=1]   --> use the J axis although it is only length 1

	DO idim = 1, nferdims
	   res_len = CX_DIM_LEN( idim, res_cx )
	   DO icom = 2, num_com
	      com_cx  = is_cx(frame_bottom+icom-1)
	      com_len = CX_DIM_LEN( idim, com_cx )
	      IF ( com_len .GT. res_len
     .	     .OR.  ( cx_lo_ww(idim,res_cx) .EQ. unspecified_val8
     .	       .AND. cx_lo_ww(idim,com_cx) .NE. unspecified_val8 ) ) THEN
	         CALL TRANSFER_AXIS( idim, com_cx, res_cx )
	         res_len = com_len
	      ENDIF
	   ENDDO
	ENDDO

* Fix up the DSG result context
* A DSG ragged array may appear as a 2D obs X E grid at this point
* If so, remove the E, as the orientation axis is the full ragged dim in length
* Doesn't apply to point data...

	IF (its_dsg .AND. orientation.LT.e_dim) THEN
	    IF (cx_hi_ss(res_cx,orientation) .NE. unspecified_int4) THEN
	       IF (cx_hi_ss(res_cx, e_dim) .NE. unspecified_int4) THEN
	    	  cx_lo_ss(res_cx, e_dim) = unspecified_int4 
	          cx_hi_ss(res_cx, e_dim) = unspecified_int4
	       ENDIF
	    ENDIF
	ENDIF
 
* allocate memory space for result
	CALL CREATE_MEM_VAR( res_cx, res_mr, status )
	IF ( status .NE. ferr_ok ) GOTO 5000
	mr_variable( res_mr ) = res_mr	! cat_temp_var vars point to themselves

* special HACK to support "a"+"b" (see MERGE_GRID)
	IF ( mr_type(res_mr) .EQ. ptype_string ) THEN
           CALL INIT_C_STRING_ARRAY( MGRID_SIZE_DELTA(res_mr),
     .                               memry(res_mr)%ptr,
     .                               mr_c_pointer(res_mr)      )
	ENDIF

* point to the memory resident components
* (if fewer than 4 components default extras to dummy)
	DO icom = 1, num_com
	   com(icom) = is_mr(frame_bottom+icom-1)
 	ENDDO
	DO icom = num_com+1, 4
	   com(icom) = dummy_mr
 	ENDDO

* set up 4D array limits as unsubscripted variables in COMMON
	CALL NON_ARRAY_SUBSC( com, 5 )

* operation relevant ONLY to numerical arguments? (neither "+" nor logical op)
	numerical_op =  .NOT.
     .	     (	(op_num.EQ.1)
     .     .OR. (op_num.GE.p_log_op_first .AND. op_num.LE.p_log_op_last)
     .     .OR. (op_num.GT.offset_logic_struc)   )

* perform the operation
	IF (its_dsg) THEN
* ... adjust the indices into 2D form for convenient DSG processing
	  CALL DSG_ARRAY_SUBSC( com, 5, orientation )
* ... good housekeeping: flag unprocessed features of the DSG result
*     to make them readily visible during code development
! DSGTBD: eventually this will be needed only in diagnostic mode
	  CALL FILL_MEMORY ( res_mr, dsg_masked_out_flag ) 
* ... get work space needed for masks
	  i8_size = nfeatures
	  CALL GET_DYN_WORK_SPACE( i8_size, status )
* ... finally do the operation
! DSGTBD:  still need DSG versions of string subroutines below
	  CALL DSG_OPERATE (op_num, num_com, ifv, res_cx, is_cx(frame_bottom),
     .			 dset,
     .			 nfeatures,
     .			 workmem(plegacy_work_buffer)%ptr,
     .			 memry(mr1)%ptr,   mr1,
     .			 memry(mr2)%ptr,   mr2,
     .			 memry(mr3)%ptr,   mr3,
     .			 memry(mr4)%ptr,   mr4,
     .			 memry(res_mr)%ptr, res_mr )

	ELSEIF (.NOT.arg_types_matched ) THEN 
* ... allow mix of float and string arguments (lets us use symbols as numbers)
	    CALL FILL_MEMORY ( res_mr, mr_bad_data(res_mr) )

	ELSEIF ( arg_type.EQ.ptype_string .AND. numerical_op ) THEN 
* ... inappropriate use of string arguments
* Issue an error instead of returning missing results on 
* inappropriate use of string arguments
	    GOTO 5200

	ELSEIF (arg_type.EQ.ptype_string) THEN
	  IF (op_num .EQ. 1 ) THEN  ! "+"
	    CALL STRING_CONCATENATE
     .		       ( is_cx(frame_bottom),
     .			 memry(mr1)%ptr,   mr1,
     .			 memry(mr2)%ptr,   mr2,
     .			 memry(res_mr)%ptr, res_mr )
	  ELSEIF (op_num .GT. offset_logic_struc) THEN
	    CALL STRING_CONDITIONAL
     .		       ( op_num-offset_logic_struc,
     .			 num_com, is_cx(frame_bottom),
     .			 memry(mr1)%ptr,   mr1,
     .			 memry(mr2)%ptr,   mr2,
     .			 memry(mr3)%ptr,   mr3,
     .			 memry(res_mr)%ptr, res_mr )
	  ELSE  ! "EQ", "NE", "LT", ...
	    CALL STRING_OPERATE
     .		       ( op_num, is_cx(frame_bottom),
     .			 memry(mr1)%ptr,   mr1,
     .			 memry(mr2)%ptr,   mr2,
     .			 memry(res_mr)%ptr, res_mr )
            cx_type(res_cx) = ptype_float
            mr_type(res_mr) = ptype_float
	  ENDIF
	ELSE
	  CALL OPERATE ( op_num, num_com, ifv, is_cx(frame_bottom),
     .			 memry(mr1)%ptr,   mr1,
     .			 memry(mr2)%ptr,   mr2,
     .			 memry(mr3)%ptr,   mr3,
     .			 memry(mr4)%ptr,   mr4,
     .			 memry(res_mr)%ptr, res_mr )
	ENDIF

* clean up 
* ... deallocate result context space
	cx_stack_ptr = cx_stack_ptr - 1

* ... if any components were un_cached we must uncache the result
	not_reusable = .FALSE.
	DO icom = isp, frame_bottom, -1
	   IF ( ITS_UN_CACHED(is_mr(icom)) ) not_reusable = .TRUE.
 	ENDDO
	IF ( not_reusable ) CALL UN_CACHE( res_mr )

* ...release component cx stack space
	DO isp = isp, frame_bottom, -1
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
! instead make the result depend on grid ABSTRACT
!* or, if the result requires
!* the same grid, then pass responsibility for deallocating it later along to
!* the result context. The use count is decremented (by calling DEALLO...)
!* for each additional component that may have shared use of this dynamic grid.
	   com_cx = is_cx(isp)
	   IF ( cx_has_impl_grid(com_cx) ) THEN
	      impl_grid = cx_grid(com_cx)
	         CALL DEALLO_DYN_GRID(impl_grid)
	      cx_has_impl_grid(com_cx) = .FALSE.
	   ENDIF
	ENDDO

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
 5100	CALL ERRMSG( ferr_invalid_command, status,
     .		'cannot use a string as an argument to: '//alg_text, *5000 )
 5200	CALL ERRMSG( ferr_invalid_command, status,
     .		'cannot use strings in numerical expression', *5000 )
	END
