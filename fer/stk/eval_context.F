	SUBROUTINE EVAL_CONTEXT ( dflt_cx, expression, status )

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
* "evaluate context of expression"
* calculate grid and completed context based on the expression given
* without actually evaluating the expression

* NOTE THAT THIS ROUTINE CAN RETURN WRONG REGION INFORMATION FOR THE SPECIAL
* case of expressions like
*	let a = XSEQUENCE(J[j=1:3])
*	`a[i=100:200]`
* where explicit limits are given that do not match returns on ABSTRACT axes
* or on enhanced-header CDF files with incomplete axis regions included


* programmer - steve hankin
* NOAA/PMEL, Seattle, WA - Tropical Modeling and Analysis Program

* V510: 3/00 - based on EVAL_EXPR

	include 'tmap_dims.parm'
	include	'ferret.parm'
	include 'errmsg.parm'
	include 'xcontext.cmn'
	include 'xvariables.cmn'
	include 'xprog_state.cmn'
	include 'xfr_grid.cmn'

* calling argument declarations
	INTEGER	dflt_cx, status
	CHARACTER*(*) expression

* local variable declarations
	INTEGER		cx_expr
	CHARACTER	EXPR_NAME*8

* initialize grid stack pointer (pushes down from top)
	grd_stk_ptr = max_grids

* delete any memory left-overs from last command
!	CALL DELETE_MRS_IN_PROGRESS

* * * * DETERMINE CONTEXT FOR COMMAND * * *
* ==> this needs to be replaced with a version that can evaluate incomplete
*	regions that have grave-accented region information
	CALL GET_NEW_CX( dflt_cx, cx_cmnd, status )
	IF ( status .NE. ferr_ok ) GOTO 5000

* store expression in the uvar list names as "EX#1"
	CALL DELETE_OLD_EXPR
	CALL BREAK_UP_EXPR( expression, num_uvars_in_cmnd, status )
	IF ( status .NE. ferr_ok ) GOTO 5000
	IF ( num_uvars_in_cmnd .NE. 1 ) CALL ERRMSG
     .			(ferr_invalid_command,status,expression,*5000)

* reserve a context slot to hold a fresh command context
	cx_stack_ptr = cx_stack_ptr + 1
	cx_expr = cx_stack_ptr

* evaluate the expression
	CALL TRANSFER_CONTEXT( cx_cmnd, cx_expr )
	CALL INTERP_CONTEXT( 1, cx_expr, EXPR_NAME(1), status )
	IF ( status .NE. ferr_ok ) GOTO 5000

! temp diagnostic
	   IF ( cx_stack_ptr .NE. cx_cmnd+1 ) WRITE (6,*) 'get_cmnd_???'

* success
	RETURN

* error exit
 5000	RETURN
	END
