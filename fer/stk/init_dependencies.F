	SUBROUTINE INIT_DEPENDENCIES
     .		( dflt_cx, yes_apply_cx, base_isp, status )

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
* this routine is derived from EVAL_EXPR -- but has only the initialization
* part from it, so that other parts of the code may be called multiple times
* for multiple expressions
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
        LOGICAL yes_apply_cx
	INTEGER	dflt_cx, base_isp, status

* local variable declarations
	INTEGER		mr, num_indices, j, uv, wkblk, total_items

* initialize grid stack pointer (pushes down from top)
	dependency_tree_mode = .TRUE.

* delete any memory left-overs from last command
	CALL DELETE_MRS_IN_PROGRESS

* * * * DETERMINE CONTEXT FOR COMMAND * * *
	CALL GET_NEW_CX( dflt_cx, cx_cmnd, yes_apply_cx, status )
	IF ( status .NE. ferr_ok ) GOTO 5000

* allocate temporary work space for recording the dependency tree in memory
	total_items = 0
	CALL deleted_list_get_undel(uvar_num_items_head, 
     .                          deleted_list_result(1),
     .                          max_uvar, num_indices )
	DO 100 j = 1, num_indices
	   uv = deleted_list_result(j)
	   total_items = total_items + uvar_num_items( uv )
 100	CONTINUE
* grossly crude heuristic to set an upper bound for the size of the completed
* structure representing the tree -- the number of elements in
* total when adding together all pyramids that can be built from a variable
* colection
* Approx half of elements are operators so divide by 2
* Upper bound size is that each operand depends on all other operands ==> n^2
* Crudely say we'd never in reality get to more than 1/4 of this
* Each record saved contains NFIELDS fields, so multiply by NFIELDS
* so ...   (500 items would will yield 93,750 words)
	dependency_max_recs = total_items/2
	dependency_max_recs = (dependency_max_recs*dependency_max_recs)/4
	dependency_max_recs = MAX(dependency_max_recs, 1000)
	CALL GET_WORK_SPC( dependency_max_recs*dependency_nfields,
     .			   wkblk, status )
	IF (status .NE. ferr_ok) GOTO 5000

* initialize dependency COMMON
	dependency_nrows      = 0
	dependency_root_level = base_isp
	dependency_block1     = wkblk
	dependency_status     = df_valid_var
	IF (cx_data_set(cx_cmnd) .EQ. unspecified_int4) THEN
	   dependency_dflt_dset  = pdset_irrelevant
	ELSE
	   dependency_dflt_dset  = cx_data_set(cx_cmnd)
	ENDIF
	RETURN

* error exit
 5000	CALL EXIT_DEPENDENCY_MODE
     	RETURN
	END
