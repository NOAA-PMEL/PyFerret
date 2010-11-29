	SUBROUTINE IS_SUBSPAN_FILL( memory, *, *, status )

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
* fill in the phantom "void" point that fills the gap between the length N
* of a native subspan modulo axis, and the virtual length N+1

* programmer - steve hankin
* NOAA/PMEL, Seattle, WA - Tropical Modeling and Analysis Program
*
* V541: *sh* 2/02
* V56: *acm* 4/04  If data was read using strides, then the axis having
*                  a delta is ok.

	include	'ferret.parm'
	include	'errmsg.parm'
	include	'interp_stack.parm'
	include	'xcontext.cmn'
	include	'xvariables.cmn'
	include	'xprog_state.cmn'
	include	'xtext_info.cmn'

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
	INTEGER CAXIS_MODLEN, MGRID_SIZE_DELTA,
     .		com_mr, res_mr, com_cx, res_cx,	com_isp, res_isp,
     .		npts, idim
	LOGICAL	TM_ITS_CDF, done_by_strides
        CHARACTER dtype*4

* --- end of introductory code ---

* initialize
	res_isp    = isp
	com_isp    = isp + 1
	res_mr     = is_mr( res_isp )
	com_mr     = is_mr( com_isp )
	res_cx     = is_cx( res_isp )
	com_cx     = is_cx( com_isp )
	idim       = is_axis( res_isp )
	npts	   = CAXIS_MODLEN( idim, com_cx )

* Check if a stride was specified on the axis.
* This is ok if data was read with strides

        done_by_strides = cx_category(res_cx) .EQ. cat_file_var 
     .    .AND. TM_ITS_CDF(cx_data_set(res_cx), dtype )
     .	  .AND. cx_delta(idim,res_cx) .NE. unspecified_val8

	IF (.NOT. done_by_strides .AND. 
     .      cx_delta(idim,res_cx) .NE. unspecified_val8) THEN
	  CALL ERRMSG( ferr_internal, status,
     .		      'IS_SUBSPAN_FILL: delta found on axis', *5900) 
	ENDIF

* diagnostic mode output: " doing subspan void VAR_NAME[x=lo:hi,D=#]"
	IF ( mode_diagnostic ) 	CALL DIAG_OP
     .		( 'doing', isact_class_subspan_fill, com_cx, idim )


* initialize the result
* 11/95 complete the result context in case it has some unspecified limits
* (note: this never occurs on file variables because INTERP_STACK completes
*	 the context to make memory caching more efficient)
	CALL RECONCILE_COM_DST_CX( com_cx, res_cx, idim )

* create context and mem. res. space if this is the first component recieved
	CALL CREATE_MEM_VAR( res_cx, res_mr, status )
	IF ( status .NE. ferr_ok ) GOTO 5000
	is_mr( res_isp ) = res_mr

	IF ( mr_type(res_mr) .EQ. ptype_string ) THEN
	   CALL INIT_C_STRING_ARRAY( MGRID_SIZE_DELTA(res_mr),
     .                               memory(1, mr_blk1(res_mr)),
     .                               mr_c_pointer(res_mr)      )
	   CALL SET_NULL_C_STRING_ARRAY(mr_c_pointer(res_mr),
     .                                  MGRID_SIZE_DELTA(res_mr) )

	ENDIF

* 11/95: if there might be voids in the result pre-fill with missing data flags
* to-do: could make this more efficient and fill only the "N+1" virtual point.
	IF ( mr_type(res_mr) .EQ. ptype_float  ) CALL FILL_MEMORY
     .				( memory,mr_blk1(res_mr),mr_nblks(res_mr),
     .                            mr_bad_data(res_mr) )

* Copy the component data into the result (unless the silly user
*  requested the void point, alone
	IF ( cx_lo_ss(res_cx, idim) .LT. npts ) CALL COPY_INTO
     .		       ( memory(1, mr_blk1(com_mr)), com_mr, com_cx,
     .			 memory(1, mr_blk1(res_mr)) ,res_mr )

* Successful completion of modulo operation
	CALL MR_NOT_IN_USE( com_mr )	! remove memory protection on component
!	IF (mr_delta(idim,com_mr).NE.unspecified_val8)
!     .		 CALL DELETE_VARIABLE( com_mr )  ! invalid since using mr_delta
	cx_stack_ptr = cx_stack_ptr - 1 ! pop com_cx  from context stack
	status = ferr_ok
	RETURN 2

* error exits
 5000	RETURN
 5900	CALL MR_NOT_IN_USE( com_mr )
	GOTO 5000
	END	
