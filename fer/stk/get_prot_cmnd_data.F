	SUBROUTINE GET_PROT_CMND_DATA ( memory, dflt_cx, status )

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
* "get protected command data"
* calculate grid(s) of data based on the expression given or implied.
* Expression(s) are in user-variable (uvar) structures.
* Implied expression is left from last command
* when done - return with the variables still "protected" (flagged as in use)

* a sample command may look like:
*	prompt> CONTOUR/Z=50 TEMP[X=160E:160W]
* the "Z=50"		is a "command  qualifier" or "command  context modifier"
* the "X=160E:160W"	is a "variable qualifier" or "variable context modifier"

* contexts are transferred, modified and used in the following manner:
*   i)	skeletal context ( without grid-related info.) is specified in the call
*	Usually it is based on SET REGION,DTA_SET,etc. ( cx_last )
*  ii)	this context is modified by the command qualifiers given to produce a 
*	new skeletal context to be used for the evaluation of the current 
*	command ( cx_cmnd )
*  iii)	as the expression is evaluated each variable, as it is encountered,
*	begins with cx_cmnd and modifies it by the variable
*	qualifiers to produce the context at which the variable is to be
*	evaluated.

* programmer - steve hankin
* NOAA/PMEL, Seattle, WA - Tropical Modeling and Analysis Program

* V230: 7/27/92 - new routine based on GET_CMND_DATA with the unprotecting
* of variables removed to a separate routine
*  (revision history for GET_CMND_DATA shown here)

*   rev 0.0 - 6/17/86
*   rev 1.0 - 1/5/87  - multiple variables in a single expression may be
*			   evaluated and pointers returned in COMMON/XCONTEXT
*   rev 2.0 - 1/5/87  - evaluations done on the INTERPRETATION STACK.  User-
*			   defined variables introduced
*   rev 2.1 - 9/16/87 - added get_cx_only to calling sequence
*			   and unknown_qual_ok moved to FERRET.PARM
*   rev 2.2 - 9/22/87 - shortened code using GET_NEW_cx
*   rev 2.3 - 9/30/87 - added mr_perm_prot
*   rev 2.4 - 7/26/88 - don't pass unstand_grid to last context
*   V200:  5/18/89 - don't pass context on to next command
*	  	   - initialize grid stack pointer
*          8/19/91 - explicit SAVE statement for last expression buffer
* V312: 5/94 - array "memory" as a calling argument
* 12/96 *kob* - Linux Port - had to move variable assignments off of
*               declaration line. used DATA statment rather than a
*	        PARAMETER statement because laslen was modified in code
*	      - Linux F90 doesn't allow SAVE statement, so put an ifdef
*		check around it 

#ifdef unix
	include 'tmap_format/tmap_dims.parm'
	include	'ferret_cmn/ferret.parm'
	include 'ferret_cmn/errmsg.parm'
	include 'ferret_cmn/xcontext.cmn'
	include 'ferret_cmn/xvariables.cmn'
	include 'ferret_cmn/xprog_state.cmn'
	include 'ferret_cmn/xfr_grid.cmn'
#else
	INCLUDE 'TMAP_FORMAT:TMAP_DIMS.PARM'
	INCLUDE	'FERRET_CMN:FERRET.PARM'
	INCLUDE 'FERRET_CMN:ERRMSG.PARM'
	INCLUDE 'FERRET_CMN:XCONTEXT.CMN'
	INCLUDE 'FERRET_CMN:XVARIABLES.CMN'
	INCLUDE 'FERRET_CMN:XPROG_STATE.CMN'
	INCLUDE 'FERRET_CMN:XFR_GRID.CMN'
#endif

* calling argument declarations
	INTEGER	dflt_cx, status
	REAL	memory(*) 

* local variable declarations
	INTEGER		cmnd_mods_cx, start, end, nu, uvar, next_start,
     .			cx_expr, mr, laslen
	DATA	 	laslen/200/
	CHARACTER	EXPR_NAME*8, last*200

* for compilers that put variables on stacks:
* not for f90 *kob* 12/96
#ifndef FORTRAN_90
        SAVE  last, laslen
#endif

* initialize grid stack pointer (pushes down from top)
	grd_stk_ptr = max_grids

* delete any memory left-overs from last command
	CALL DELETE_MRS_IN_PROGRESS

* * * * DETERMINE CONTEXT FOR COMMAND * * *
	CALL GET_NEW_CX( dflt_cx, cx_cmnd, status )
	IF ( status .NE. ferr_ok ) GOTO 5000

* * * * BREAK FULL EXPRESSION INTO SUB_EXPRESSIONS * * *
* ... and store them in the uvar list names as "EX#n"
* ... eg. LIST U^2,V^2 breaks into "U^2" and "V^2"
* ... or re-use the last expression ?
	IF ( num_args .GT. 0 ) THEN
	   IF ( last(:laslen) .NE. cmnd_buff(arg_start(1):len_cmnd) 
     .	   .OR. num_uvars_in_cmnd .EQ. cmnd_uvars_not_given 		) THEN
	      last = cmnd_buff(arg_start(1):len_cmnd)
	      laslen = len_cmnd - arg_start(1) + 1
	      CALL DELETE_OLD_EXPR
	      CALL BREAK_UP_EXPR( cmnd_buff( arg_start(1):len_cmnd ),
     .				  num_uvars_in_cmnd, status )
	      IF ( status .NE. ferr_ok ) THEN
	         num_uvars_in_cmnd = cmnd_uvars_not_given
	         GOTO 5000
	      ENDIF
	   ENDIF
	ENDIF

* * * * EVALUATE THE SUB-EXPRESSIONS * * *
	IF ( num_uvars_in_cmnd .EQ. cmnd_uvars_not_given )
     .		CALL ERRMSG
     .			(ferr_invalid_command,status,'no data specified',*5000)

	DO 300 nu = 1, num_uvars_in_cmnd

* ... a context slot to hold a fresh command context for each sub-expression
	   cx_stack_ptr = cx_stack_ptr + 1
	   cx_expr = cx_stack_ptr

* ... evaluate the next sub-expression
	   CALL TRANSFER_CONTEXT( cx_cmnd, cx_expr )
	   CALL INTERP_STACK( memory, nu, cx_expr,
     .			      EXPR_NAME(nu), status )
	   IF ( status .NE. ferr_ok ) GOTO 5000

! temp diagnostic
	   IF ( cx_stack_ptr .NE. cx_cmnd+nu ) WRITE (6,*) 'get_cmnd_???'
 300	CONTINUE

* turn off oversizing flag for next command
	cx_ovsiz(dflt_cx) = ovsiz_off
	RETURN

* error exit
 5000	RETURN
	END
