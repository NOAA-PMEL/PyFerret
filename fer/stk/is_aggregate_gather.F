	SUBROUTINE IS_AGGREGATE_GATHER( memory, *, *, status )

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
* bring the components of an aggregate defined with DEFINE DATA/AGGREGATE 
* together by requesting the members one by one on the interpretation stack

* programmer - Ansley Manke
* NOAA/PMEL, Seattle, WA - Thermal Modeling and Analysis Program
*
* V680 5/12 *acm* 6D Ferret, changes for DEFINE DATA/AGGREGATE
* V69+ 5/14 *acm* Fixes for ticket 2177: datasets with _FillValue, no missing_value
* V695 2/15 *acm* 6D add aggregation in forecast direction
* V698 1/16 *sh* fix to bug #2347 (res var no instead of com var on stack)
*		altered logic as described in header of INIT_EF_AGGREGATE_DSET

	include	'tmap_dims.parm'
	include	'ferret.parm'
	include	'errmsg.parm'
	include	'interp_stack.parm'
	include	'xcontext.cmn'
	include	'xvariables.cmn'
	include	'xprog_state.cmn'

* calling argument declarations:
	INTEGER	status
	REAL	memory( mem_blk_size, max_mem_blks )
* normal return is for error conditions
* exit #1 is to request another member ( stack has been pushed )
* exit #2 is to signal completion of the activity at this stack level

*   ... interpretation stack setup at the time this is called ...
*	stack level		stack contents
*	isp+1	<--	level of last activity ( component just computed )
*	 isp	<--	current stack level (for gathering )

* internal variable declarations:
	LOGICAL	NC_GET_ATTRIB, NC_GET_ATTRIB_STRING, TM_SAME_GRID_DEF,
     .		attoutflag, got_it, itsa_dummy
	INTEGER NCF_GET_ATTR_FROM_ID, TM_LENSTR1, STR_SAME, KNOWN_GRID,
     .		com_mr, res_mr, com_cx, res_cx,	com_isp, res_isp,
     .		agg_dim, top, imemb, memb_dset, iline,
     .		varid, attlen, maxlen, agg_dset, slen, vlen,
     .		orig_cat, orig_var, orig_grid,
     .		 com_cat,  com_var,  com_grid

	CHARACTER*128 VAR_CODE, vname, buff
	CHARACTER LEFINT*6

* --- end of introductory code ---

* initialize
	res_isp    = isp
	com_isp    = isp + 1
	res_mr     = is_mr( res_isp )
	com_mr     = is_mr( com_isp )
	res_cx     = is_cx( res_isp )
	com_cx     = is_cx( com_isp )
	agg_dim    = is_axis( res_isp )
        agg_dset   = cx_data_set(res_cx)

* diagnostic mode output: " doing aggregate gather VAR_NAME[x=lo:hi@TRN:n,D=#]"
* Put in the agg- index value just for this output

	IF ( mode_diagnostic ) THEN
	   cx_lo_ss( com_cx,agg_dim ) = is_phase(isp)  ! hack to make
	   cx_hi_ss( com_cx,agg_dim ) = is_phase(isp)  ! diagnostic informative
	   CALL DIAG_OP
     .	    ( 'doing',isact_class_aggregate_gather, com_cx, agg_dim )
	   cx_lo_ss( com_cx,agg_dim ) = unspecified_int4
	   cx_hi_ss( com_cx,agg_dim ) = unspecified_int4
	ENDIF

* create mem. result from result cx if this is the first component received
	IF ( res_mr .EQ. unspecified_int4 ) THEN
	   CALL CREATE_MEM_VAR( res_cx, res_mr, status )
	   IF ( status .NE. ferr_ok ) GOTO 5000
	   is_mr( res_isp ) = res_mr
	ENDIF

* copy data just received into the result
* bad data flag differences are resolved during this COPY operation
	CALL COPY_AGG_INTO( memory(1,mr_blk1(com_mr)), com_mr, com_cx, agg_dim,
     .			memory(1,mr_blk1(res_mr)), res_mr, is_phase(isp) )

* ... remove the component protection that signaled we needed it here
	IF (cx_data_set(com_cx) .NE. pdset_dummy) THEN
	   CALL MR_NOT_IN_USE( com_mr )
	ELSE
	   CALL DELETE_VARIABLE( com_mr )   ! delete array of missing vals
	ENDIF

* are we done ?
	IF ( is_phase(isp) .EQ. is_sum(isp) ) THEN
* ... yes - pop com_cx from context stack
	   cx_stack_ptr = cx_stack_ptr - 1
	   RETURN 2
	ENDIF

**************************************
* NOT DONE -- set up com_cx to get the next aggregate member
	is_phase(res_isp) = is_phase(res_isp) + 1

! *sh* as seen in earlier version ...
! these parts of the component context were already set in IS_STRIP
!	CALL TRANSFER_CONTEXT( big_cx, com_cx )

	imemb = is_phase(res_isp)

	vname = VAR_CODE (cat_file_var, cx_variable(res_cx))   ! *sh* 1/16
	vlen = TM_LENSTR1(vname)
	CALL CD_GET_AGG_VAR_INFO (agg_dset, vname, imemb, 
     .      orig_cat, memb_dset, orig_grid, iline, orig_var, status)

* Is it the 2D Time variable for a Forecast-aggregation?  If so then
* there has been a cache hit failure (FIND_MEM_VAR_INCL_C_CACHE)
* ... should never happen
	IF (memb_dset .EQ. pagg_promotion_indicator_dset)
     .		CALL ERRMSG(ferr_internal, status,
     .		'aggregation failure in is_aggregate_gather', *5000)

	itsa_dummy = memb_dset .EQ. pdset_dummy
	IF (itsa_dummy) THEN
* ... for a dummy (missing-filled) dset, borrow context info of the 1st member
	   CALL CD_GET_AGG_VAR_INFO (agg_dset, vname, 1, 
     .         orig_cat, memb_dset, orig_grid, iline, orig_var, status)
	ENDIF

* do a fresh "find" for the variable name, to allow for the possibility that
* LET/D variables have been defined, canceled or changed since DEFINE DATA/AGG
	CALL FIND_VAR_NAME (memb_dset,vname, com_cat, com_var )
	IF (com_var .EQ. munknown_var_name) CALL ERRMSG(
     .		ferr_aggregate_error, status, 'variable '//
     .		vname(:vlen)//' no longer found in agg member '//
     .		LEFINT(imemb,slen), *5000)

* grid must not have changed since DEFINE DATA/AGG
	com_grid = KNOWN_GRID(memb_dset, com_cat, com_var)
	IF (.NOT.TM_SAME_GRID_DEF(com_grid,orig_grid)) CALL ERRMSG(
     .		ferr_aggregate_error, status,
     .		'coordinate grid of '// vname(:vlen)//
     .		' has changed in agg member '//LEFINT(imemb,slen), *5000)

* Unspecified limits on agg-direction are retained from previous component
!	CALL DEL_CX_DIM(agg_dim, com_cx)   ! in IS_STRIP
!	cx_by_ss(agg_dim,com_cx) = .TRUE.  ! in IS_STRIP

* the T axis of an F-aggregation has a single set of indices but the time
* values differ for each axis.  So re-interpret the time range from the indices
* (Indices are retained throughout from result cx)
	IF (agg_dim .EQ. f_dim) THEN
	   cx_by_ss(t_dim, com_cx) = .TRUE.
	   CALL FLESH_OUT_AXIS(t_dim, com_cx, status )
	   IF (status .NE. ferr_ok) GOTO 5000
	ENDIF

* set up the member variable
	cx_data_set(com_cx) = memb_dset
	cx_category(com_cx) = com_cat
	cx_variable(com_cx) = com_var
	cx_grid    (com_cx) = com_grid
	CALL GET_CONTEXT_BAD_FLAG(com_cx)
	
	IF (itsa_dummy) THEN
* dummy dataset triggers a missing data fill operation in INTERP_STACK
	   cx_data_set(com_cx) = pdset_dummy
	ENDIF

	is_uvar(com_isp) = com_var ! relevant if component 1 is LET/D

* push the interpretation stack and go get it
	isp = com_isp
	status = ferr_ok
	RETURN 1

* error exit
 5000	RETURN
	END	
