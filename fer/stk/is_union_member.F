	SUBROUTINE IS_UNION_MEMBER(  *, *, status )

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
* return the union member just read.

* programmer - Ansley Manke
* NOAA/PMEL, Seattle, WA - SDIG
*
* V7   5/12 *acm* Union aggregations: DEFINE DATA/AGG/U

        IMPLICIT NONE
	include	'tmap_dims.parm'
	include	'ferret.parm'
	include	'errmsg.parm'
	include	'interp_stack.parm'
	include	'xcontext.cmn'
	include	'xvariables.cmn'
	include	'xprog_state.cmn'
	include 'xtm_grid.cmn_text'   ! for grid_line

* calling argument declarations:
	INTEGER	status
* normal return is for error conditions
* exit #2 is to signal completion of the activity at this stack level

*   ... interpretation stack setup at the time this is called ...
*	stack level		stack contents
*	isp+1	<--	level of last activity ( component just computed )
*	 isp	<--	current stack level (for gathering )

* internal variable declarations:
	INTEGER NCF_GET_ATTR_FROM_ID, 
     .		com_mr, res_mr, com_cx, res_cx, com_isp, res_isp,
     .		res_grid, res_cat, res_var, imemb, memb_dset, iline,
     .		varid, attlen, maxlen, agg_dset, slen

	CHARACTER*128 VAR_CODE, agg_vname, memb_vname, buff
	CHARACTER LEFINT*6

* --- end of introductory code ---

* initialize
	res_isp    = isp
	com_isp    = isp + 1
	res_mr     = is_mr( res_isp )
	com_mr     = is_mr( com_isp )
	res_cx     = is_cx( res_isp )
	com_cx     = is_cx( com_isp )
        agg_dset   = cx_data_set(res_cx)
	res_grid   = cx_grid(res_cx)
	res_cat    = cx_category(res_cx)
	res_var    = cx_variable(res_cx)

* diagnostic mode output: " doing aggregate gather VAR_NAME[x=lo:hi@TRN:n,D=#]"
* Put in the agg- index value just for this output

	IF ( mode_diagnostic ) THEN
	   CALL DIAG_OP
     .	    ( 'doing', isact_get_union_member, com_cx, no_dim )
	ENDIF

* create mem. result from result cx 
	IF ( res_mr .EQ. unspecified_int4 ) THEN
	   CALL CREATE_MEM_VAR( res_cx, res_mr, status )
	   IF ( status .NE. ferr_ok ) GOTO 5000
	   is_mr( res_isp ) = res_mr
	ENDIF

* copy data just received into the result
* bad data flag differences are resolved during this COPY operation
	CALL COPY_INTO( memry(com_mr)%ptr, com_mr, com_cx,
     .			memry(res_mr)%ptr, res_mr )

* ... remove the component protection that signaled we needed it here
	IF (cx_data_set(com_cx) .NE. pdset_dummy) THEN
	   CALL MR_NOT_IN_USE( com_mr )
* ... delete it altogether (avoid memory clogging) unless its in use elsewhere
	   IF (mr_protected(com_mr) .EQ. mr_not_protected) 
     .					CALL DELETE_VARIABLE( com_mr )
	ELSE
	   CALL DELETE_VARIABLE( com_mr )   ! delete dummy (missing vals) 
	ENDIF

* we are done 
* ... yes - pop com_cx from context stack

	IF ( is_phase(isp) .EQ. is_sum(isp) ) THEN
	   cx_stack_ptr = cx_stack_ptr - 1
	   RETURN 2
	ENDIF

**************************************


* error exit
 5000	RETURN
	END	
