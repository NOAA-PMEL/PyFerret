	SUBROUTINE IS_REVERSE( memory, *, *, status )

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
* reverse the ordering of the data along the axes indicated in ds_ordering
* and delete the (backwards) component data from memory

* programmer - steve hankin
* NOAA/PMEL, Seattle, WA - Tropical Modeling and Analysis Program
*
* V320: 2/16/95
* V411: 9/11/95 (IBM port) array "reversed" is declared wrong type
* V500: 7/99 *sh* is strides were applied to reversed axes need special
*	handling of the limits
* V541: 2/02 *sh* - bug fix -- 6/01 changes in the handling of delta strides
*			(see read_tm.F) were not propagated to this routine

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
*	 isp	<--	current stack level (for moduloing )

* internal variable declarations:
	INTEGER TM_AXES_REVERSED, nrev,
     .		com_mr, res_mr, com_cx, res_cx,	com_isp, res_isp,
     .		axis, i, var, hi_ss, lo_ss, itmp, reversed(4)

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

!* strides indicated in cx_delta e.g. SST[I=110:180:10] ?
!* If so, transfer this info to the stride array
!        has_a_stride = .FALSE.
!        DO 100 idim = 1, 4
!           has_strides(idim) =  cx_delta(idim,res_cx) .NE. unspecified_val8
!     .                   .AND.  cx_delta(idim,res_cx) .NE. 1 
!           IF (has_strides(idim))  has_a_stride = .TRUE.
! 100    CONTINUE

!* if strides are invloved memory requirement is less ... (cloned from READ_TM)
!* 2/02 -- no longer necessary or appropriate
!        CALL TRANSFER_CONTEXT(res_cx, cx_buff)
!        IF (has_a_stride) THEN
!           DO 200 idim = 1, 4
!              IF  (has_strides(idim)) THEN
! 1  	cx_lo_ss( cx_buff, idim ) = 1 
!               cx_hi_ss( cx_buff, idim ) = 
!     .                (cx_hi_ss(res_cx,idim) - cx_lo_ss(res_cx,idim))
!     .              / INT(cx_delta(idim,res_cx))  + 1
!                 cx_delta(idim,res_cx) = unspecified_val8
!              ENDIF
! 200      CONTINUE
!        ENDIF

* create the space for the result in memory
	CALL CREATE_MEM_VAR( res_cx, res_mr, status )
	IF ( status .NE. ferr_ok ) GOTO 5100
	is_mr( res_isp ) = res_mr

* the component data thinks that its subscripts are from "the other end" of
* the axis (from whence the data were read).  In order to prevent subscript
* errors we will disabuse the data of that misconception atthis point
	DO 300 i = 1, nrev
	   axis = reversed(i)
!	   IF (has_strides(axis)) GOTO 300	  ! already handled above
	   CALL VAR_SS_LIMS( axis, res_cx, lo_ss, hi_ss )
	   itmp = mr_lo_ss( com_mr, axis )
	   mr_lo_ss(com_mr,axis) = hi_ss + lo_ss - mr_hi_ss(com_mr,axis)
	   mr_hi_ss(com_mr,axis) = hi_ss + lo_ss - itmp
	   itmp = cx_lo_ss( com_cx, axis )
	   cx_lo_ss(com_cx,axis) = hi_ss + lo_ss - cx_hi_ss(com_cx,axis)
	   cx_hi_ss(com_cx,axis) = hi_ss + lo_ss - itmp
 300	CONTINUE

* copy the source data into the result
	CALL COPY_INTO(
     .    memory( 1, mr_blk1(com_mr) ), com_mr, com_cx,
     .    memory( 1, mr_blk1(res_mr) ), res_mr   )

* reverse the result along all of the required axes
	DO 400 i = 1, nrev
	   CALL REVERSE_GRID_AXIS
     .		( reversed(i), res_cx, res_mr,memory(1, mr_blk1(res_mr)) )
 400	CONTINUE

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

