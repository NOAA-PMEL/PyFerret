	SUBROUTINE IS_MODULO( memory, *, *, status )

*
*
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
* inserting pieces obtained by non-modulo reads (within the base axis limits)

* programmer - steve hankin
* NOAA/PMEL, Seattle, WA - Tropical Modeling and Analysis Program
*
* V320: 19/9/94
* V420: 11/95 - added RECONCILE_COM_DST_CX for use in modulo regridding
*	      - auto-modulo (stripped off in INTERP_STACK for all classes
*		of variables) can no longer assume that the component data
*		lies strictly in the base region
*	      - If the result is "pathological" then flag it as "un-cached"
*		so it will not be reused.

#ifdef unix
	include	'ferret_cmn/ferret.parm'
	include	'ferret_cmn/errmsg.parm'
	include	'ferret_cmn/interp_stack.parm'
	include	'ferret_cmn/xcontext.cmn'
	include	'ferret_cmn/xvariables.cmn'
	include	'ferret_cmn/xprog_state.cmn'
	include	'ferret_cmn/xtext_info.cmn'
#else
	INCLUDE	'FERRET_CMN:FERRET.PARM'
	INCLUDE	'FERRET_CMN:ERRMSG.PARM'
	INCLUDE	'FERRET_CMN:INTERP_STACK.PARM'
	INCLUDE	'FERRET_CMN:XCONTEXT.CMN'
	INCLUDE	'FERRET_CMN:XVARIABLES.CMN'
	INCLUDE	'FERRET_CMN:XPROG_STATE.CMN'
	INCLUDE	'FERRET_CMN:XTEXT_INFO.CMN'
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
	INTEGER CAXIS_LEN, CX_DIM_LEN,
     .		com_mr, res_mr, com_cx, res_cx,	com_isp, res_isp,
     .		npts, idim, i, phase, res_lo, res_hi, xlate, nmod,
     .		tmp_mr, nrepl, rqst_cx


* --- end of introductory code ---

* this routine attempt to replicate **whatever** component data has been
* supplied to form a modulo result.  Normally the component data will
* lie within the base range, 1 to Npts, (length of modulo cycle).  However,
* in odd cases the component data may be anywhere and of any size.  This
* routine will base its modulo output on the first Npts of the component
* data.  It will go back to get further component data only if instructed
* to do so by IS_STRIP (i.e. phase==isphase_1)


* |++++++++|--------|--------|--------|--------|   <-- replicated modulo axis
*                              |-------------|	   <-- available component data
*               |----------------|		   <-- requested region

* Examples of the odd cases would be
*   LIST/i=50:60 v5[i=30:40]    where v5 is on a 5 point modulo.
*   (The incoming component to modulo i=50:60 will be i=30:40.)
* or
*   LET v = temp[g=gmod,L=big:really_big]  ! regrid non-modulo to a modulo axis
*   LIST/L=1:48 v
*   (The incoming component is non-modulo and located at big L values.) 

* The special case in which IS_STRIP will supply data in two phases is when
* a small modulo result is requested that crosses a modulo boundary:
* |++++++++|--------|--------|--------|--------|   <-- replicated modulo axis
*                |-----|		   	   <-- requested region
*                 ^ | ^
*	Phase:    1   2

* initialize
	res_isp    = isp
	com_isp    = isp + 1
	res_mr     = is_mr( res_isp )
	com_mr     = is_mr( com_isp )
	res_cx     = is_cx( res_isp )
	rqst_cx    = is_big_cx( res_isp )
	com_cx     = is_cx( com_isp )
	idim       = is_axis( res_isp )
	npts	   = CAXIS_LEN( idim, com_cx )
	phase	   = is_phase(isp)

* diagnostic mode output: " doing modulo VAR_NAME[x=lo:hi,D=#]"
	IF ( mode_diagnostic ) 	CALL DIAG_OP
     .				( 'doing',isact_class_modulo, com_cx, idim )

	IF (.NOT.(phase.EQ.isphase_allatonce
     .	     .OR. phase.EQ.isphase_1.OR.phase.EQ.isphase_2) ) CALL ERRMSG
     .		(ferr_internal, status, 'modulo_phase_err', *5000)

	IF ( phase .NE. 2 ) THEN

* 11/95 complete the result context in case it has some unspecified limits
* (note: this never occurs on file variables because INTERP_STACK completes
*	 the context to make memory caching more efficient)
	   CALL RECONCILE_COM_DST_CX( com_cx, res_cx, idim )

* create context and mem. res. space if this is the first component recieved
	   CALL CREATE_MEM_VAR( res_cx, res_mr, status )
	   IF ( status .NE. ferr_ok ) GOTO 5000
	   is_mr( res_isp ) = res_mr

* 11/95: if there might be voids in the result pre-fill with missing data flags
	   IF ( CX_DIM_LEN(idim,com_cx) .LT. npts ) CALL FILL_MEMORY
     .				( memory,mr_blk1(res_mr),mr_nblks(res_mr),
     .                            mr_bad_data(res_mr) )
	ENDIF

* If the component received (com_cx) does not match the component requested
* (rqst_cx) then we can proceed with the modulo but other calculations
* should not reuse it as it may have voids.  For example, if npts=10 then
* the results of "list/i=1:15 v1[i=3:7]" should not be reused.
	IF (cx_lo_ss(com_cx,idim) .NE. cx_lo_ss(rqst_cx,idim)
     . .OR. cx_hi_ss(com_cx,idim) .NE. cx_hi_ss(rqst_cx,idim) ) 
     .						CALL UN_CACHE(res_mr)

* The internals of this routine require a dual look at the components that
* are recieved:  in the view that arrives with the components they are located
* within the base region (non-modulo, I=1:npts) of the variable.  In order to
* copy into the result, however, they must also be viewable in the subscript
* range(s) of the result.  For this purpose dual memory table entries are
* created that will describe the component data.  The dual version will be
* deleted before exiting this routine.
	CALL FIND_MR_SLOT( com_cx, tmp_mr, status )
	IF ( status .NE. ferr_ok ) GOTO 5910
	CALL ASSIGN_MR_CX( com_cx, tmp_mr )     ! effects diag output, only
	DO 100 i = 1, 4
	   mr_lo_ss(tmp_mr,i) = mr_lo_ss(com_mr,i)
	   mr_hi_ss(tmp_mr,i) = mr_hi_ss(com_mr,i)
 100	CONTINUE
	mr_blk1 (tmp_mr) = mr_blk1 (com_mr)    ! just for good house-keeping
	mr_nblks(tmp_mr) = mr_nblks(com_mr)    ! just for good house-keeping
	mr_protected(tmp_mr) = mr_table_entry_only

* * * * *
* Copy the component data into the result, replicating as we go
	nrepl = (cx_hi_ss(res_cx,idim) - cx_lo_ss(res_cx,idim))
     .		   / npts
	res_hi = cx_lo_ss(res_cx,idim) - 1
	DO 300 i = 1, nrepl + 1
	   res_lo = res_hi + 1
	   res_hi = MIN( res_lo+npts-1, cx_hi_ss(res_cx,idim) )
	   CALL SHORT_MOD_COPY( memory,
     .				res_lo, res_hi, idim,
     .				com_cx, tmp_mr, res_cx, res_mr)
 300	CONTINUE
* * * * *

* Set up context to go back for data that is still needed
	IF( phase .EQ. isphase_1 ) THEN
* ... update the original rqst_cx to make new request
*     note: the variable EX#N may have become category file variable in com_cx 
	   nmod = cx_hi_ss(res_cx, idim) / npts
	   xlate = nmod * npts
	   cx_lo_ss(rqst_cx, idim) = 1
	   cx_hi_ss(rqst_cx, idim) = cx_hi_ss(res_cx, idim) - xlate
	   cx_by_ss(idim,rqst_cx) = .TRUE.
	   CALL FLESH_OUT_AXIS( idim, rqst_cx, status )
	   IF ( status .NE. ferr_ok ) GOTO 5900

* ... set up the request for the next component 
	   CALL TRANSFER_CONTEXT( rqst_cx, com_cx )

* ... remove the component protection that signaled we needed it here
	   CALL MR_NOT_IN_USE( com_mr )
	   CALL DELETE_VARIABLE( tmp_mr )  ! delete the extra table entry

* ... push the interpretation stack and go get the final component
	   is_phase(isp) = isphase_2
	   isp = com_isp
	   status = ferr_ok
	   RETURN 1

* Successful completion of modulo operation
	ELSE
	   CALL MR_NOT_IN_USE( com_mr )	! remove memory protection on component
	   CALL DELETE_VARIABLE( tmp_mr )  ! delete extra table entry
	   cx_stack_ptr = cx_stack_ptr - 1 ! pop rqst_cx from context stack
	   cx_stack_ptr = cx_stack_ptr - 1 ! pop com_cx  from context stack
	   status = ferr_ok
	   RETURN 2
	ENDIF

* error exits
 5000	RETURN
 5900	CALL DELETE_VARIABLE( tmp_mr )  ! delete extra table entry
 5910	CALL DELETE_VARIABLE( res_mr )  ! delete invalid result
	CALL MR_NOT_IN_USE( com_mr )
	GOTO 5000
	END	
