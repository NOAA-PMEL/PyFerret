	SUBROUTINE IS_GATHER( memory, *, *, status )

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
* bring together a full data grid from fragmentary pieces by requesting those
* pieces one by one on the interpretation stack

* programmer - steve hankin
* NOAA/PMEL, Seattle, WA - Tropical Modeling and Analysis Program
* written for VAX computer under VMS operating system
*
* V200:  6/8/89
* V312: 5/94 - array "memory" as a calling argument

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
*	 isp	<--	current stack level (for gathering )

* internal variable declarations:
	INTEGER com_mr, res_mr, com_cx, res_cx,	dflt_cx, com_isp, res_isp,
     .		frag, idim, top


* --- end of introductory code ---

* initialize
	res_isp    = isp
	com_isp    = isp + 1
	res_mr     = is_mr( res_isp )
	com_mr     = is_mr( com_isp )
	res_cx     = is_cx( res_isp )
	com_cx     = is_cx( com_isp )
	dflt_cx	   = is_big_cx( res_isp )
	idim       = is_axis( res_isp )
	frag	   = is_sum ( res_isp )

* diagnostic mode output: " doing gather VAR_NAME[x=lo:hi@TRN:n,D=#]"
	IF ( mode_diagnostic ) 	CALL DIAG_OP
     .				( 'doing',isact_class_gather, com_cx, idim )

* create context and mem. res. space if this is the first component recieved
	IF ( res_mr .EQ. unspecified_int4 ) THEN
* ... use gathering axis from dflt_cx and everything else from component
* ... (since all else may be altered during evaluation if its a user var)
	   CALL TRANSFER_CONTEXT( com_cx, res_cx )
	   CALL TRANSFER_AXIS( idim, dflt_cx, res_cx )
	   CALL CREATE_MEM_VAR( res_cx, res_mr, status )
	   IF ( status .NE. ferr_ok ) GOTO 5000
	   is_mr( res_isp ) = res_mr
	ENDIF

* copy data just received into the result
	CALL COPY_INTO( memory(1,mr_blk1(com_mr)), com_mr, com_cx,
     .			memory(1,mr_blk1(res_mr)), res_mr )

* ... remove the component protection that signaled we needed it here
	CALL MR_NOT_IN_USE( com_mr )

* are we done ?
	IF ( cx_hi_ss(com_cx,idim) .EQ. cx_hi_ss(dflt_cx,idim) ) THEN
* ... yes - pop com_cx and dflt_cx from context stack
	   IF ( dflt_cx .NE. cx_stack_ptr ) STOP 'GATH_STK_ERR'	! temp
	   cx_stack_ptr = cx_stack_ptr - 2
	   RETURN 2
	ENDIF

* no - set up com_cx to get the next component grid
	top = cx_hi_ss(com_cx,idim)
	CALL TRANSFER_CONTEXT( dflt_cx, com_cx )
	cx_lo_ss(com_cx,idim) = top + 1
	cx_hi_ss(com_cx,idim) = MIN( top+frag, cx_hi_ss(dflt_cx,idim)	)
	cx_by_ss(idim,com_cx) = .TRUE.
	CALL FLESH_OUT_AXIS( idim, com_cx, status )
	IF ( status .NE. ferr_ok ) GOTO 5000

* push the interpretation stack and go get it
	isp = com_isp
	status = ferr_ok
	RETURN 1

* error exit
 5000	RETURN
	END	

