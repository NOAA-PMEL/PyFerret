	SUBROUTINE IS_SAMPLE( memory, *, *, status )

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
* This routine handles the I/O management logic for the SAMPLE* functions
* When the user issues SAMPLEI({1,5,3,3,8},my_var) this routine will
* compute the result in an "optimal" way -- probably by requesting 1:5 in
* one I/O operation and 8:8 in a separate one and the merging the chunks

* programmer - steve hankin
* NOAA/PMEL, Seattle, WA - Tropical Modeling and Analysis Program
*
* V510: 8/99
*	 *sh* 2/00 (??) - correct improper handling of invalid indices
* V533; *sh* 7/01 - add support for sampling of string variables

	include	'ferret.parm'
	include	'errmsg.parm'
	include	'interp_stack.parm'
	include	'xcontext.cmn'
	include	'xvariables.cmn'
	include	'xprog_state.cmn'

* calling argument declarations:
	INTEGER	status
	REAL	memory( mem_blk_size, max_mem_blks )
* normal return is for error conditions
* exit #1 is to request another grid ( stack has been pushed )
* exit #2 is to signal completion of the activity at this stack level

*   ... interpretation stack setup at the time this is called ...
*	stack level		stack contents
*	isp+1	<--	level of last activity ( component just computed )
*	 isp	<--	current stack level (for sampling )

* internal variable declarations:
	LOGICAL	all_at_once
	INTEGER DENSE_CONTIG_REGION, MGRID_SIZE_DELTA
	INTEGER chunk_mr, hold_mr, chunk_cx, hold_cx,
     .		axis, n, phase, lo_cx_lim, hi_cx_lim, mr_sorted_indices,
     .		mr_index_map, idim, chunk_isp, hold_isp, cat, var,
     .		nvalid


* --- end of introductory code ---

* initialize
	hold_isp   = isp
	chunk_isp  = isp + 1
	hold_mr    = is_mr( hold_isp )		! to be filled by chunks
	chunk_mr   = is_mr( chunk_isp )		! each invidivual chunk
	hold_cx    = is_cx( hold_isp )
	chunk_cx   = is_cx( chunk_isp )
!	dflt_cx	   = is_big_cx( hold_isp )
	axis       = is_axis( hold_isp )
	phase      = is_phase(hold_isp)
	mr_index_map      = is_sum(hold_isp)/10000
	mr_sorted_indices = is_sum(hold_isp) - 10000*mr_index_map
	nvalid = mr_hi_ss( mr_index_map, t_dim )	! valid indices
	cat = cx_category(chunk_cx)
	var = cx_variable(chunk_cx)

****  A NEW CHUNK OF DATA HAS JUST ARRIVED  ****
*  Copy it into the result "holder"
	IF ( phase .GT. 0 ) THEN

* the context for the holder may be missing its limits on axes other than the
* sampling axis
	   DO 100 idim = 1, 4
	      IF ( idim .NE. axis ) THEN
	         CALL TRANSFER_AXIS( idim, chunk_cx, hold_cx )
	      ENDIF
 100	   CONTINUE

* create result space if this is the first chunk
	   IF ( hold_mr .EQ. unspecified_int4 ) THEN
	      CALL CREATE_TEMP_MEM_VAR( hold_cx, hold_mr, status )
	      IF ( status .NE. ferr_ok ) GOTO 5000
	      is_mr( hold_isp ) = hold_mr

* pre-fill output with missing flags, in case a missing index was given
	      IF ( mr_type(hold_mr) .EQ. ptype_string ) THEN
	        CALL INIT_C_STRING_ARRAY( MGRID_SIZE_DELTA(hold_mr),
     .                                    memory(1, mr_blk1(hold_mr)),
     .                                    mr_c_pointer(hold_mr)      )
	        CALL SET_NULL_C_STRING_ARRAY(mr_c_pointer(hold_mr),
     .					     MGRID_SIZE_DELTA(hold_mr) )

	      ELSE
	        CALL FILL_MEMORY ( memory, mr_blk1 (hold_mr),
     .				   mr_nblks(hold_mr),
     .				   mr_bad_data(hold_mr) )
	      ENDIF

	   ENDIF

	   IF ( nvalid .GT. 0 ) THEN
* for optimization, make sure we are using all of the source data that we can
* the ADJUST_COPY_LIMITS routine will modify "phase" to use all available data
	     n = phase
	     CALL ADJUST_COPY_LIMITS(axis, chunk_mr,
     .				     memory(1,mr_blk1(mr_sorted_indices)),
     .				     phase, nvalid )
	     IF (phase .GT. n) THEN
* ... apply the optimization
	        cx_hi_ss(chunk_cx, axis) = mr_hi_ss(chunk_mr,axis)
	        cx_by_ss(axis,chunk_cx) = .TRUE.
	        CALL FLESH_OUT_AXIS( axis, chunk_cx, status)
	        IF (status .NE. ferr_ok) CALL ERRMSG( ferr_internal, status,
     .		    'SAMPLE* optimization', *5000)
	     ENDIF

* copy data just received into the result
	     CALL COPY_INTO_MAPPED(
     .			memory(1,mr_blk1(chunk_mr)), chunk_mr, chunk_cx,
     .			memory(1,mr_blk1(mr_sorted_indices)),
     .			memory(1,mr_blk1(mr_index_map)), phase, axis,
     .			memory(1,mr_blk1(hold_mr)), hold_mr )

* ... remove the component protection that signaled we needed it here
	   ENDIF
	   CALL MR_NOT_IN_USE( chunk_mr )
	ENDIF


*********  SETUP TO GRAB THE NEXT CHUNK  **********

* do we want to grab the entire input as a single sample?
	IF (cat .EQ. cat_user_var) THEN
* ... if limits are part of the uvar def'n, use 'em
* ...  and if an indefinite integral is involved, do not break up computation
	   all_at_once = uvar_given(axis,var) .LE. uvlim_gvn_xact
     .		    .OR. uvar_given(axis,var) .EQ. uvlim_need_lo
	ELSE
	  all_at_once = cat .EQ. cat_const_var
     .	           .OR. cat .EQ. cat_pseudo_var
     .	           .OR. cat .EQ. cat_constant
	ENDIF
* ... if limits are explicit inside of the fcn call then use 'em
	all_at_once = all_at_once .OR. cx_given(axis,chunk_cx)

* total number of indices
	n = mr_hi_ss( mr_index_map, x_dim )

* determine next chunk to get
	IF ( nvalid .EQ. 0 ) THEN
	   IF (phase .EQ. 0) THEN
	      lo_cx_lim = 1	! get any old point just so we can have
	      hi_cx_lim = 1	! a complete context for the result
	      is_phase(hold_isp) = 1
	   ELSE
	      is_phase(hold_isp) = -1  ! all done with sampling
	   ENDIF   
	ELSEIF ( all_at_once ) THEN
	   IF (phase .EQ. 0) THEN
	      is_phase(hold_isp) = nvalid
	   ELSE
	      is_phase(hold_isp) = -1  ! all done with sampling
	   ENDIF
	   lo_cx_lim = cx_lo_ss(chunk_cx,axis)  ! grab all of it at once
	   hi_cx_lim = cx_hi_ss(chunk_cx,axis)
	ELSE
	   is_phase(hold_isp) = DENSE_CONTIG_REGION(
     .				memory(1,mr_blk1(mr_sorted_indices)), 
     .				phase+1, nvalid,
     .				mr_bad_data(mr_sorted_indices),
     .				lo_cx_lim, hi_cx_lim )
	ENDIF

* done?  Pop chunk_cx from context stack
	IF ( is_phase(hold_isp) .LE. 0 ) THEN
	   cx_stack_ptr = cx_stack_ptr - 1
	   IF ( hold_cx .NE. cx_stack_ptr ) CALL WARN('SAMP_STK_ERR')
* ... free the work space used for the sampling inddices
	   CALL DELETE_VARIABLE(mr_index_map)
	   CALL DELETE_VARIABLE(mr_sorted_indices)
	   RETURN 2
	ENDIF

* no, more to get.  Set up chunk_cx to get the next slug of indices
	cx_lo_ss(chunk_cx,axis) = lo_cx_lim 
	cx_hi_ss(chunk_cx,axis) = hi_cx_lim
	cx_by_ss(axis,chunk_cx) = .TRUE.
	CALL FLESH_OUT_AXIS( axis, chunk_cx, status )
	IF ( status .NE. ferr_ok ) GOTO 5000

* push the interpretation stack and go get it
	isp = chunk_isp
	status = ferr_ok
	RETURN 1

* error exit
 5000	RETURN

	END	

