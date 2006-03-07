	SUBROUTINE EVAL_EXPR ( memory, dflt_cx, expression, apply_cx, 
     .              status )

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
* "evaluate expression"
* calculate grid of data based on the expression given
* Saves (temporarily) expression in user-variable (uvar) structure.
* when done - return with the variable un-"protected" (flagged as deletable)

* a sample command may look like:
*	CONTOUR/Z=`TEMP[Z=0:1000@LOC:10]` TEMP[X=160E:160W]

* contexts are transferred, modified and used in the following manner:
*   i)	skeletal context ( without grid-related info.) is specified in the call
*	Usually it is based on SET REGION,DTA_SET,etc. ( cx_last )
*  ii)	this context is modified by the command qualifiers given to produce a 
*	new skeletal context to be used for the evaluation of the current 
*	command ( cx_cmnd )
*	the command context may, itself, contain grave-accented expressions
*	... those parts of the region are ignored until they are known
*  iii)	as the expression is evaluated each variable, as it is encountered,
*	begins with cx_cmnd and modifies it by the variable
*	qualifiers to produce the context at which the variable is to be
*	evaluated.

* programmer - steve hankin
* NOAA/PMEL, Seattle, WA - Tropical Modeling and Analysis Program

* V400: 3/95 - based on GET_PROT_CMND_DATA
* V590:  3/07 *acm* - fixing bug 439&1390. Add logical variable its_action_command 
*                     to decide when the qualifiers /X=/Y=/Z=/T= apply to the evaluation
*                     of immed. mode expressions in the command.
* V590 *acm* 3/06  fixing bugs 439&1390, pass apply_cx to get_new_cx to say whether
*                  to apply command context to grave-accent expressions.

	include 'tmap_dims.parm'
	include	'ferret.parm'
	include 'errmsg.parm'
	include 'xcontext.cmn'
	include 'xvariables.cmn'
	include 'xprog_state.cmn'
	include 'xfr_grid.cmn'


* calling argument declarations
        LOGICAL apply_cx
	INTEGER	dflt_cx, status
	REAL	memory(*) 
	CHARACTER*(*) expression

* local variable declarations
	INTEGER		cx_expr, mr
	CHARACTER	EXPR_NAME*8
	INTEGER		tlen

* initialize grid stack pointer (pushes down from top)
	tlen = LEN(expression)
	grd_stk_ptr = max_grids

* delete any memory left-overs from last command
	CALL DELETE_MRS_IN_PROGRESS

* * * * DETERMINE CONTEXT FOR COMMAND * * *
* ==> this needs to be replaced with a version that can evaluate incomplete
*	regions that have grave-accented region information
	CALL GET_NEW_CX( dflt_cx, cx_cmnd, apply_cx, status )
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
	CALL INTERP_STACK( memory, 1, cx_expr, EXPR_NAME(1), status )
	IF ( status .NE. ferr_ok ) GOTO 5000

! temp diagnostic
	   IF ( cx_stack_ptr .NE. cx_cmnd+1 ) WRITE (6,*) 'get_cmnd_???'

* flag this result as temporary/delete-able
        CALL UNPROTECT_CMND_DATA
	RETURN

* error exit
 5000	RETURN
	END
