	SUBROUTINE IS_RECONCILE( memory, *, *, status )

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
* reconcile any conflicts between the component requested and the one received
* for example, LIST/I=3:7  test[i=2:5]

* programmer - steve hankin
* NOAA/PMEL, Seattle, WA - Tropical Modeling and Analysis Program
*
* V420: 11/95

#ifdef unix
	include	'ferret.parm'
	include	'errmsg.parm'
	include	'interp_stack.parm'
	include	'xcontext.cmn'
	include	'xvariables.cmn'
	include	'xprog_state.cmn'
	include	'xtext_info.cmn'
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
	LOGICAL	by_ss(4), given(4)
	INTEGER i, mr, com_cx, res_cx,	com_isp, res_isp,
     .		com_lo, com_hi, res_lo(4), res_hi(4)
	CHARACTER VAR_CODE*24, qual*1

* --- end of introductory code ---

* initialize
	res_isp    = isp		! result (what the user requested)
	com_isp    = isp + 1		! component (what we have received)
	res_cx     = is_cx( res_isp )
	com_cx     = is_cx( com_isp )
	mr	   = is_mr( com_isp )

* defaults
	is_mr( res_isp ) = mr
	is_cx( res_isp ) = res_cx

* save the key ingredients from the requesting context (res_cx)
* (FLESH_OUT_AXIS has already been applied so world coords need not be used)
	DO 10 i = 1, 4
	   res_lo(i) =  cx_lo_ss(res_cx,i)
	   res_hi(i) =  cx_hi_ss(res_cx,i)
	   by_ss(i)  =  cx_by_ss(i,res_cx)
	   given(i)  =  cx_given(i,res_cx)
 10	CONTINUE

* transfer the full context (which includes data set, etc.)
	CALL TRANSFER_CONTEXT( com_cx, res_cx )

* check each axis for possible reconciliation
	DO 100 i = 1, 4
	   IF ( given(i) ) THEN
	      com_lo = cx_lo_ss(com_cx,i)
	      com_hi = cx_hi_ss(com_cx,i)
	      IF ( com_lo    .EQ. unspecified_int4
     .	      .OR. res_lo(i) .EQ. unspecified_int4 ) GOTO 100
	      IF (res_lo(i).NE.com_lo .OR. res_hi(i).NE.com_hi) THEN

* unreconcilable limits
	         IF (res_lo(i).GT.com_hi .OR. res_hi(i).LT.com_lo) THEN
	            IF ( by_ss(i) ) THEN
	               qual = ss_dim_name(i) 
	            ELSE
	               qual = ww_dim_name(i) 
	            ENDIF
	            CALL WARN('Invalid '//qual//
     .			' axis limits ignored: '//
     .			 VAR_CODE(cx_category(res_cx),cx_variable(res_cx)) )

	         ELSE
* diagnostic mode output: " doing reconcile VAR_NAME[x=lo:hi,D=#]"
	            IF ( mode_diagnostic ) 	CALL DIAG_OP
     .				( 'doing',isact_class_reconcile, com_cx, i )

* trim to limits that satisfy both component and result
	            cx_lo_ss(res_cx,i) = MAX(com_lo, res_lo(i))
	            cx_hi_ss(res_cx,i) = MIN(com_hi, res_hi(i))
	            cx_by_ss(i, res_cx) = .TRUE.
	            CALL FLESH_OUT_AXIS(i, res_cx, status )
	            IF ( status .NE. ferr_ok ) GOTO 5000
	         ENDIF
	      ENDIF
	   ENDIF
 100	CONTINUE

* success -- clean up
	cx_stack_ptr = cx_stack_ptr - 1 ! pop com_cx  from context stack
	CALL MR_NOT_IN_USE( mr )	! remove memory protection on component
	RETURN 2

* error exits
 5000	CALL MR_NOT_IN_USE( mr )	! remove memory protection on component
	RETURN
	END	
