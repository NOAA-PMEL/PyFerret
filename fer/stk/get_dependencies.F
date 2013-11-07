	SUBROUTINE GET_DEPENDENCIES( uvar, dset, work, status )
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
* this routine is derived from GET_UVAR_GRID
* see that routine for a discussion of how it is implemented
* instead of evaluating the grid of a uvar, this routine traces through the
* same path and reports out the dependencies

* programmer - steve hankin
* NOAA/PMEL, Seattle, WA - Tropical Modeling and Analysis Program
*
* V685+ 9/13 *sh*  

        include 'tmap_errors.parm'
        include 'tmap_dims.parm'
        include 'xtm_grid.cmn_text'
        external xgt_grid_data

	include	'ferret.parm'
	include	'interp_stack.parm'
	include	'errmsg.parm'
	include	'xcontext.cmn'
	include	'xvariables.cmn'
	include	'xdependency_tree.cmn'

* calling argument declarations:
	INTEGER	uvar, dset, status
	INTEGER	work(dependency_max_recs, dependency_nfields)

* internal variable declarations:
	INTEGER	isp_orig, cx_orig, cx, idim

* pre-load the variable info onto the CX stack
	cx_orig = cx_stack_ptr
	CALL STACK_PTR_UP( cx_stack_ptr, max_context, status )
	IF ( status .NE. ferr_ok ) GOTO 5000
	cx = cx_stack_ptr
	cx_data_set(cx) = dset
	cx_variable(cx) = uvar
	cx_category(cx) = cat_user_var

* pre-load the variable info onto the IS stack
	isp_orig = isp
	CALL PUSH_INTERP_STACK( isp, status )
	IF ( status .NE. ferr_ok ) GOTO 5000
	is_cx   ( isp ) = cx
	is_uvar ( isp ) = uvar
	is_act  ( isp ) = isact_algebra	! temp diagnostic

* initialize the stacks - (repeated for each component)
 100	is_phase( isp ) = 0
	is_sum  ( isp ) = 0   ! flag to change after getting aux vars
	IF ( status .NE. ferr_ok ) GOTO 5000

* evaluate the dependencies through recursive descent
*  ... exit 100 is to evaluate the dependencies for a component uvar
*  ... exit 300 means that a variable in need of reporting has been obtained
*  ... "normal" return indicates an error has occurred
* calling this starts a new dependency tree -- create a forward link from last
 200	CALL IS_DEPENDENCIES( *100, *300, work, status )
	GOTO 5000
 
* a uvar has just been reported.  Are we done ?
 300	IF ( isp .EQ. isp_orig ) THEN
* ... yes - clean up and go home
	   IF ( cx_stack_ptr .NE. cx_orig ) WRITE (6,*) 'cx_stk_lev_crptn'
	   RETURN
	ELSE
* ... no - go back for more
	   GOTO 200
	ENDIF

* error exit
 5000	IF ( status .EQ. ferr_stack_ovfl )
     .		CALL WARN( 'Check for recursion in variable definitions' )
	RETURN
	END
