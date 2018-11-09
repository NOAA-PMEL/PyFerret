	SUBROUTINE IS_AGGREGATE_GATHER(  *, *, status )

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
*      3/16 *acm*always mark aggregation components unused after using them,
*      3/16 *sh* delete agg component after using if nothig else has a claim
*      4/16 *sh* Allow that SET VAR/NAME= has altered the agg varname
* V741 *acm* 5/18 Fixes so string-variables are returned in aggregations

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
* exit #1 is to request another member ( stack has been pushed )
* exit #2 is to signal completion of the activity at this stack level

*   ... interpretation stack setup at the time this is called ...
*	stack level		stack contents
*	isp+1	<--	level of last activity ( component just computed )
*	 isp	<--	current stack level (for gathering )

* internal variable declarations:
	LOGICAL	NC_GET_ATTRIB, NC_GET_ATTRIB_STRING, TM_SAME_GRID_DEF,
     .		SCALAR_GRID_LINE,
     .		attoutflag, got_it, itsa_dummy
	INTEGER NCF_GET_ATTR_FROM_ID, TM_LENSTR1, STR_SAME, KNOWN_GRID,
     .		MGRID_SIZE_DELTA,
     .		com_mr, res_mr, com_cx, res_cx, com_isp, res_isp,
     .		res_grid, res_cat, res_var,
     .		agg_dim, top, imemb, memb_dset, iline, idim,
     .		varid, attlen, maxlen, agg_dset, slen, agg_vlen, memb_vlen,
     .		orig_cat, orig_var, orig_grid,
     .		 com_cat,  com_var,  com_grid

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
	agg_dim    = is_axis( res_isp )
        agg_dset   = cx_data_set(res_cx)
	res_grid   = cx_grid(res_cx)
	res_cat    = cx_category(res_cx)
	res_var    = cx_variable(res_cx)

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


	   IF ( mr_type(res_mr) .EQ. ptype_string ) THEN
	      CALL INIT_C_STRING_ARRAY( MGRID_SIZE_DELTA(res_mr),
     .                               memry(res_mr)%ptr,
     .                               mr_c_pointer(res_mr)      )
	      CALL SET_NULL_C_STRING_ARRAY(mr_c_pointer(res_mr),
     .                                     MGRID_SIZE_DELTA(res_mr) )
	   ENDIF

	   is_mr( res_isp ) = res_mr
	ENDIF

* copy data just received into the result
* bad data flag differences are resolved during this COPY operation
	CALL COPY_AGG_INTO( memry(com_mr)%ptr, com_mr, com_cx, agg_dim,
     .			memry(res_mr)%ptr, res_mr, is_phase(isp) )

* ... remove the component protection that signaled we needed it here
	IF (cx_data_set(com_cx) .NE. pdset_dummy) THEN
	   CALL MR_NOT_IN_USE( com_mr )
* ... delete it altogether (avoid memory clogging) unless its in use elsewhere
	   IF (mr_protected(com_mr) .EQ. mr_not_protected) 
     .					CALL DELETE_VARIABLE( com_mr )
	ELSE
	   CALL DELETE_VARIABLE( com_mr )   ! delete dummy (missing vals) 
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

	agg_vname = VAR_CODE (res_cat, res_var)   ! *sh* 1/16
	agg_vlen = TM_LENSTR1(agg_vname)
	CALL CD_GET_AGG_VAR_INFO (agg_dset, agg_vname, imemb, 
     .      orig_cat, memb_dset, orig_grid, iline, orig_var, status)

* for special case (hack) of an E-aggregation of FMRCs, the 2dtime array comes
* from one particular member.  We have just completed that, so we're done
	IF (orig_cat .EQ. cat_ef_time2d) THEN
* ... pop com_cx from context stack
	   cx_stack_ptr = cx_stack_ptr - 1
	   RETURN 2
	ENDIF

* Is it the 2D Time variable for a Forecast-aggregation?  If so then
* there has been a cache hit failure (FIND_MEM_VAR_INCL_C_CACHE)
* ... should never happen
	IF (memb_dset .EQ. pagg_promotion_indicator_dset)
     .		CALL ERRMSG(ferr_internal, status,
     .		'aggregation failure in is_aggregate_gather', *5000)

	itsa_dummy = memb_dset .EQ. pdset_dummy
	IF (itsa_dummy) THEN
* ... for a dummy (missing-filled) dset, borrow context info of the 1st member
	   CALL CD_GET_AGG_VAR_INFO (agg_dset, agg_vname, 1, 
     .         orig_cat, memb_dset, orig_grid, iline, orig_var, status)
	ENDIF

* a couple of 'delayed mode' use cases to beware -- actions the user may have
* taken **SINCE** the aggregation was defined:
*  1) user may have issued a SET VAR/NAME= (rename) of the aggregated variable
*  2) user may have issued a LET/D on a member variable
* If 1, then agg_vname will differ from memb_vname 
* If 2, then cat# and var# associated with member name may have changed
* Do a fresh "find" for the original member variable name
	IF (orig_cat .NE. cat_agg_gpromote) THEN
	   memb_vname = VAR_CODE(orig_cat, orig_var)
	   memb_vlen = TM_LENSTR1(memb_vname)
	   CALL FIND_VAR_NAME (memb_dset,memb_vname, com_cat, com_var )
	   IF (com_var .EQ. munknown_var_name) CALL ERRMSG(
     .		ferr_aggregate_error, status, 'variable '//
     .		memb_vname(:MEMB_vlen)//' no longer found in agg member '//
     .		LEFINT(imemb,slen), *5000)
	ELSE
	   com_cat = orig_cat
	   com_var = orig_var
	ENDIF

* grid must not have changed since DEFINE DATA/AGG
	com_grid = KNOWN_GRID(memb_dset, com_cat, com_var)
	IF (com_grid .EQ. unspecified_int4) THEN   ! can only be a uvar
* ... GET_UVAR_GRID uses the interp_stack, so bump isp
	   isp = com_isp
	   CALL GET_UVAR_GRID( com_var, memb_dset, status )
	   isp = res_isp  ! restore to value at start of routine
           IF ( status .NE. ferr_ok ) GOTO 5000
	   com_grid = KNOWN_GRID(memb_dset, com_cat, com_var)
	ENDIF

* check that component grid is conformable with aggregation
* (grid may have changed due to new LET/D definition)
	CALL CONFORMABLE_AGG_GRID(agg_dim, memb_vname, imemb,
     .                            res_grid, res_cat, res_var,
     .                            com_grid, com_cat, com_var,
     .                            orig_grid, status )
	IF (status .NE. ferr_ok) GOTO 5000

* scalar axes of components need not match the agg result
	DO idim = 1, nferdims
! 4/16	   IF ( idim.NE.agg_dim  .AND.    ! would not allow E-agg of E-aggs
	   IF (SCALAR_GRID_LINE(idim, com_grid, com_cat, com_var) ) THEN
	      CALL DEL_CX_DIM(idim, com_cx) 
	   ENDIF
	ENDDO

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
