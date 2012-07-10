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
* bring together a full data grid from an aggregate defined with DEFINE DATA/AGGREGATE 
* by requesting the members one by one on the interpretation stack

* programmer - Ansley Manke
* NOAA/PMEL, Seattle, WA - Thermal Modeling and Analysis Program
*
* V680 5/12 *acm* 6D Ferret, changes for DEFINE DATA/AGGREGATE

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
	LOGICAL	NC_GET_ATTRIB, attoutflag, got_it
	INTEGER NCF_GET_ATTR_FROM_ID,
     .		com_mr, res_mr, com_cx, res_cx,	dflt_cx, com_isp, res_isp,
     .		frag, idim, top, imemb, memb_dset, len, igrid, iline,
     .		varid, attlen, maxlen, fer_uvar, nv
	REAL	vbad
	CHARACTER*128 VAR_CODE, vname
	CHARACTER*10 buff

* --- end of introductory code ---

* initialize
	res_isp    = isp
	com_isp    = isp + 1
	res_mr     = is_mr( res_isp )
	com_mr     = is_mr( com_isp )
	res_cx     = is_cx( res_isp )
	com_cx     = is_cx( com_isp )
	dflt_cx	   = is_big_cx( res_isp )
	idim       = is_axis( res_isp )
	frag	   = is_sum ( res_isp )

* diagnostic mode output: " doing aggregate gather VAR_NAME[x=lo:hi@TRN:n,D=#]"
* Put in the M index value just for this output

	IF ( mode_diagnostic ) THEN
	   cx_lo_ss( com_cx,idim ) = is_phase(isp)
	   cx_hi_ss( com_cx,idim ) = is_phase(isp)
	   CALL DIAG_OP
     .				( 'doing',isact_class_aggregate_gather, com_cx, idim )
	   cx_lo_ss( com_cx,idim ) = unspecified_int4
	   cx_hi_ss( com_cx,idim ) = unspecified_int4
	ENDIF

* create context and mem. res. space if this is the first component recieved
	IF ( res_mr .EQ. unspecified_int4 ) THEN
* ... use gathering axis, category, grid and dataset from dflt_cx (aggregate
* ... variable, datset and grid), and everything else from component
* ... (since all else may be altered during evaluation if its a user var)
	   CALL TRANSFER_CONTEXT( com_cx, res_cx )

	   CALL TRANSFER_AXIS( idim, dflt_cx, res_cx )
	   cx_grid(res_cx) = cx_grid(dflt_cx)
	   cx_data_set(res_cx) = cx_data_set(dflt_cx)
	   cx_bad_data(res_cx) = cx_bad_data(dflt_cx)
	   cx_category(res_cx) = cx_category(dflt_cx)

	   CALL CREATE_MEM_VAR( res_cx, res_mr, status )
	   IF ( status .NE. ferr_ok ) GOTO 5000
	   is_mr( res_isp ) = res_mr
	ENDIF

* copy data just received into the result
	CALL COPY_AGG_INTO( memory(1,mr_blk1(com_mr)), com_mr, com_cx,
     .			memory(1,mr_blk1(res_mr)), res_mr, is_phase(isp) )

* ... remove the component protection that signaled we needed it here
	CALL MR_NOT_IN_USE( com_mr )

* are we done ?
	IF ( is_phase(isp) .EQ. is_sum(isp) ) THEN
* ... yes - pop com_cx and dflt_cx from context stack

	   IF ( dflt_cx .NE. cx_stack_ptr ) STOP 'AGG_GATH_STK_ERR'	! temp

	   cx_stack_ptr = cx_stack_ptr - 2
	   RETURN 2
	ENDIF

* no - set up com_cx to get the next aggregate member
	is_phase(res_isp) = is_phase(res_isp) + 1
	CALL TRANSFER_CONTEXT( dflt_cx, com_cx )
* Set limits on e-direction to be unspecified
 
	cx_lo_ss( com_cx,idim ) = unspecified_int4
	cx_hi_ss( com_cx,idim ) = unspecified_int4
	cx_by_ss( idim,com_cx) = .TRUE.

* Set the grid to the grid of the next aggregate component...

	imemb = is_phase(res_isp)

	vname = VAR_CODE (cat_file_var, cx_variable(com_cx))
	CALL CD_GET_AGG_VAR_INFO (cx_data_set(com_cx), vname, imemb, 
     .      fer_uvar, memb_dset, igrid, iline, nv, status)

* Grid wasnt stored for this variable if its a LET/D variable in member dset
* Let it be the agg grid ?? (or could set the grid to a number less than 
* unspecified_int4 to call get_uvar_grid for this LET/D member var)

	IF (igrid .EQ. unspecified_int4) THEN 
	   cx_data_set(com_cx) = memb_dset
	   cx_category(com_cx) = cat_user_var
	   cx_variable(com_cx) = nv
	ELSE

	   CALL CD_GET_VAR_ID (memb_dset, vname, varid, status)
	   maxlen = 1
	   got_it = NC_GET_ATTRIB( memb_dset, varid, 'missing_value',
     .                .FALSE., vname, maxlen, attlen,
     .                attoutflag, buff, vbad )
           IF (.NOT.got_it) vbad = bad_val4

	   cx_grid( com_cx ) = igrid
	   cx_data_set(com_cx) = memb_dset

	   IF (fer_uvar .EQ. 3) THEN
	      cx_category(com_cx) = cat_user_var  
	      cx_bad_data(com_cx) = cx_bad_data(dflt_cx)
	   ELSE
	      cx_category(com_cx) = cat_file_var 
	      cx_bad_data(com_cx) = vbad
	   ENDIF


	   CALL FLESH_OUT_AXIS( idim, com_cx, status )
	   IF ( status .NE. ferr_ok ) GOTO 5000
	ENDIF  ! or end this above flesh_out??
	
	is_uvar(com_isp) = cx_variable(com_cx)

* push the interpretation stack and go get it
	isp = com_isp
	status = ferr_ok
	RETURN 1

* error exit
 5000	RETURN
	END	
