	SUBROUTINE CRAWL_DEPENDENCIES
     .		(  expression, base_cx, status )

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
* this routine is derived from EVAL_EXPR, much split off into INIT_DEPENDENCIES
* see that routine for a discussion of how it is implemented
* instead of evaluating the expression, it determines the dependencies in it

* base_isp, base_cx allow the calling program to set base levels for work on
* the stack, allowing pre-existing stack contents to be protected

* programmer - steve hankin
* NOAA/PMEL, Seattle, WA - Tropical Modeling and Analysis Program
*
* V685+ 9/13 *sh*  

	include 'tmap_dims.parm'
	include	'ferret.parm'
	include 'errmsg.parm'
	include 'xcontext.cmn'
	include 'xvariables.cmn'
	include 'xdependency_tree.cmn'

* calling argument declarations
	INTEGER	base_cx, status
	CHARACTER*(*) expression

* local variable declarations
	INTEGER		UVAR_NUMBER, uvar_pointer, i
	CHARACTER	EXPR_NAME*8

* initialize

* store expression in the uvar list names as "EX#1"
	CALL DELETE_OLD_EXPR
	CALL BREAK_UP_EXPR( expression, num_uvars_in_cmnd, status )
	IF ( status .NE. ferr_ok ) GOTO 5000

	DO 100 i = 1, num_uvars_in_cmnd
* set up the stacks to look at this expression
	   uvar_pointer = UVAR_NUMBER( EXPR_NAME(i) )  ! (easier way to do this?)
           IF ( uvar_pointer .EQ. unspecified_int4 ) CALL ERRMSG
     .             ( ferr_internal, status, 'crawl_dep', *5000 )
	   cx_stack_ptr = base_cx
	   CALL TRANSFER_CONTEXT( cx_cmnd, base_cx )
           cx_variable( base_cx ) = uvar_pointer
           cx_category( base_cx ) = cat_user_var
           isp = dependency_root_level     ! set stack pointer to base
           is_cx  ( isp ) = base_cx        ! context for stack activities
           is_uvar( isp ) = uvar_pointer   ! pointer to text buffer

* start a new dependency tree
	   dependency_current_tree_start = dependency_nrows + 1
	
* when called the request is ALWAYS for an untransformed implicitly defined
           CALL GET_DEPENDENCIES( uvar_pointer,
     .		dependency_dflt_dset, workmem(plegacy_work_buffer)%ptr,
     .		status )
           IF ( status .NE. ferr_ok ) GOTO 5000

! temp diagnostic
	   IF ( cx_stack_ptr .NE. base_cx ) WRITE (6,*) 'get_expr_dep???'
 100	CONTINUE

	RETURN

* error exit
 5000	CALL EXIT_DEPENDENCY_MODE
     	RETURN
	END
