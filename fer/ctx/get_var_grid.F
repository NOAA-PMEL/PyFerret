	INTEGER FUNCTION GET_VAR_GRID
     .		      ( full_var_name, 
     .			default_cx,
     .			var_cx,
     .			status )

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
* given a full variable specification of the form:
*			(eg.) TEMP[X=170E:170W:2]
* return the grid, only, of the result

* this routine extracted (by quick hack) from GET_VAR_CONTEXT so that GC
* variables can determine their grids in advance of determining inheritance
* for their arguments

* V510: 1/00 *sh**
	
#ifdef unix
	include	'ferret.parm'
	include	'errmsg.parm'
	include	'xcontext.cmn'
	include	'xvariables.cmn'
#else
	INCLUDE	'FERRET_CMN:FERRET.PARM'
	INCLUDE	'FERRET_CMN:ERRMSG.PARM'
	INCLUDE	'FERRET_CMN:XCONTEXT.CMN'
	INCLUDE	'FERRET_CMN:XVARIABLES.CMN'
#endif

* calling argument declarations:
	INTEGER		default_cx, var_cx, status
	CHARACTER*(*)	full_var_name

* internal variable declarations:
	INTEGER		mods_cx

* initialize
	CALL TRANSFER_CONTEXT ( default_cx, var_cx )

* find a context slot for the modifications that may be attached to the var name
	CALL STACK_PTR_UP ( cx_stack_ptr, max_context, status )
	IF ( status .NE. ferr_ok ) GOTO 5000
	mods_cx = cx_stack_ptr
	CALL INIT_CONTEXT_MODS ( mods_cx )

* get variable and attached modifications
	CALL PARSE_VAR_NAME ( full_var_name, var_cx, mods_cx, status )
	IF ( status .NE. ferr_ok ) GOTO 5000

* 8/99 If G=unknown_user_var was given then exit now.
* It is recognizable because the text position is encoded in the grid number.
* This should happen ONLY when this is called from IS_UVAR_GRID
	IF ( cx_unstand_grid(var_cx)
     . .AND. cx_grid(var_cx) .LT. unspecified_int4 ) GOTO 5100

* determine the defining grid
	CALL GET_CONTEXT_GRID ( var_cx, status )
	IF ( status .NE. ferr_ok ) GOTO 5000

* can't do more if the grid isn't known
	IF ( cx_grid(var_cx) .EQ. unspecified_int4 ) GOTO 5100

* relinquish the stack space used for the modifications
 100	CALL STACK_PTR_DN ( cx_stack_ptr, cx_stack_ptr_base, status )
	IF ( status .NE. ferr_ok ) GOTO 5000
	RETURN

* error exit
 5000	RETURN
 5100	CALL ERRMSG( ferr_unknown_grid, status, full_var_name, *5000 )
	END
