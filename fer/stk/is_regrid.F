	SUBROUTINE IS_REGRID( memory, *, status )

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
* regrid thae data from the source grid to the destination grid

* programmer - steve hankin
* NOAA/PMEL, Seattle, WA - Tropical Modeling and Analysis Program
* written for VAX computer under VMS operating system
*
* revision 0.0 - 7/25/88
* revision 0.1 -12/30/88 - eliminated P1 and Q1 from regrid call
*			 - renamed DO_REGRID --> DO_LIN_REGRID
* revision 0.2 -  1/9/89 - added xy average regridding
* V200:  8/2/89 - 4D symmetrical version - usually 1 axis per regrid
*	10/16/89 - set up adjustable array limits with NON_ARRAY_SUBSC
*	 11/3/89 - if destination limits are unknown use full axis length
*	 12/1/89 - allow XY average regridding - code split to RCVD_REGRID
* V230:  8/22/92 - added @ASN and @AVE regridding
* V300:  3/16/93 - allow regridding where the source range falls entirely
*                  between points on the destination axis (all bad_data result)
* V312: 5/94 - array "memory" as a calling argument
* V420: 9/95	 - Support for implicit grids: deallocate after it is used
*      11/95	 - added modulo (climatological) regridding
*      12/95	 - added modngd, modmin, etc. (climatological) regridding
*	1/96	 - bug check against attempt to regrid a compressed axis
*	1/29/96	 - regridding @SUM and @NGD are not weighted
*		   (done in ave_regrid_1_axis instead of window_regrid_1_axis)
* V491: 10/97 *sh* - add G=@XACT
* V500:  5/99 *sh* - check if implicit regridding was already done by netCDF
*			read using strides (check cx_delta of src cx)
*		     Allow for the possibility of multiple axes
* V530:  3/01 *sh* - added g=ax@nrst
* V530:	 3/01 *acm* return a status flag from rcvd_regrid, tests for
*                   valid calendar axes.
* V533:	 6/01 *sh* - some regridding ops not allowed for string arrays
*		   - mr_delta changes the handling of stride regridding
* v541:  2/02 *kob*- on hankin request add an initialization line for
*                    cx_delta in loop 500

	include	'ferret.parm'
	include	'errmsg.parm'
	include	'interp_stack.parm'
	include	'xprog_state.cmn'
	include	'xcontext.cmn'
	include	'xvariables.cmn'
	include	'xalgebra.cmn'   ! with alg_regrid_code

* calling argument declarations:
	INTEGER	status
	REAL	memory( mem_blk_size, max_mem_blks )
* normal exit is for error conditions
* exit #1 is to signal completion of the activity at this stack level

* internal variable declarations:
	LOGICAL	TM_ITS_CDF,
     .		xy_ave, valid, valid_y, has_impl_save, done_by_strides,
     .		itsa_string
	INTEGER hi_isp, lo_isp, axis, cx_valid, idim,
     .		rgrd_trans, mrlist(4), wk_size, wk_size2,
     .		dst_cx, src_cx, dst_mr, src_mr, ngd_mr, ave_mr,
     .		blk1, axis_list(4), naxis, i
	CHARACTER dtype*4

*        ... interpretation stack setup at the time this is called ...
*	stack level		stack contents
*	isp+2	<--	level of next activity
*	hi_isp	<--	level at which component was obtained
*	lo_isp	<--	level at which regrid was requested (current isp)
* --- end of introductory code ---

* initialize
	lo_isp   = isp                     
	hi_isp   = lo_isp + 1
	src_mr   = is_mr( hi_isp )
	dst_cx   = is_cx( lo_isp )
	src_cx   = is_cx( hi_isp )
	axis	 = is_axis( lo_isp )
	naxis	 = 1
	axis_list(1) = axis
	itsa_string = cx_type(src_cx) .EQ. ptype_string

* what is the regridding transformation for this axis ?
	rgrd_trans = cx_regrid_trans( axis, dst_cx )
	xy_ave = rgrd_trans .EQ. prgrd_xy_ave
	IF ( xy_ave ) THEN
	   IF (axis.NE.x_dim) CALL ERRMSG
     .		( ferr_internal, status, 'rgrd_xy_ave', *5000 )	! diagnostic
	   naxis = 2
	   axis_list(2) = y_dim
	ENDIF

* bug check:  has someone attempted to regrid a compressed axis?
* (possible bug in check ... could we have unspec limits for another reason?)
	IF ( cx_category(src_cx) .EQ. cat_user_var
     . .AND. cx_lo_ss(src_cx,axis) .EQ. unspecified_int4 ) THEN
	   IF (  uvar_given(axis,cx_variable(src_cx))
     .	    .NE. uvlim_irrelevant )			 CALL ERRMSG
     .		( ferr_invalid_command, status,
     .		 'attempt to regrid on a compressed axis', *5000 )
	ENDIF

* (5/99) for netCDF variables strides may have been used in READ_TM and
* the regridding already performed there. The src cx then contains the
* limits appropriate to netCDF ... not to Ferret. So, make the source cx
* into a clone of the final result.
	IF ( cx_category(dst_cx) .EQ. cat_file_var ) THEN
	   done_by_strides = 
     .		TM_ITS_CDF(cx_data_set(dst_cx), dtype )
     .    .AND. cx_has_impl_grid(dst_cx)
     .	  .AND. cx_delta(axis,src_cx) .NE. unspecified_val8
	ELSE
	   done_by_strides = .FALSE.
	ENDIF
	IF (done_by_strides) THEN
	   DO 400 idim = axis+1, 4
	      IF (cx_delta(idim,src_cx) .NE. unspecified_val8) THEN
	         naxis = naxis + 1
	         axis_list(naxis) = idim
	      ENDIF
 400	   CONTINUE
	ENDIF

* Make the destination context like the source data context on all except
* the regrid axis.  That one keeps the user-requested size.
* (Other axes may have been modified by information embedded within a
* user-variable definition.  An alternative (untested 8/89) approach is to
* call PASS_AMBIG_LI** for uvars)
	CALL TRANSFER_CONTEXT( src_cx, cx_buff )
	DO 500 i = 1, naxis 
	   idim = axis_list(i)
	   CALL TRANSFER_AXIS( idim, dst_cx, cx_buff )
	   cx_regrid_trans( idim, cx_buff ) = cx_regrid_trans( idim, dst_cx )
	   cx_delta(idim,cx_buff) = unspecified_val8
 500	CONTINUE
	cx_grid( cx_buff ) = cx_grid( dst_cx )
	cx_unstand_grid( cx_buff ) = .TRUE.
	has_impl_save = cx_has_impl_grid(dst_cx)
	CALL TRANSFER_CONTEXT( cx_buff, dst_cx )
	cx_has_impl_grid(dst_cx) = has_impl_save	! preserve this flag

* create a context with the limits that can actually be regridded
* ... if limits were unspecified in request allow full axis size
	IF (done_by_strides) THEN
	   cx_valid = dst_cx
	   valid = .TRUE.
	ELSE
	   cx_valid = cx_buff
	   CALL RCVD_REGRID( src_cx, dst_cx, axis, wk_size,
     .		 cx_lo_ss(cx_valid,axis), cx_hi_ss(cx_valid,axis), status )
	   cx_by_ss( axis, cx_valid ) = .TRUE.
           valid = cx_lo_ss(cx_valid,axis) .LE. cx_hi_ss(cx_valid,axis)  ! 3/93
           IF ( valid ) THEN   ! 3/93
	      CALL CONFINE_AXIS( axis, cx_valid, status )
	      IF ( status .NE. ferr_ok ) GOTO 5000
           ENDIF
	   IF ( xy_ave ) THEN
	      CALL RCVD_REGRID( src_cx, dst_cx, y_dim, wk_size2,
     .		cx_lo_ss(cx_valid,y_dim),cx_hi_ss(cx_valid,y_dim), status)
	      cx_by_ss( y_dim, cx_valid ) = .TRUE.
              valid_y = cx_lo_ss(cx_valid,y_dim)
     .             .LE. cx_hi_ss(cx_valid,y_dim)
              IF ( valid_y ) THEN
	         CALL CONFINE_AXIS( y_dim, cx_valid, status )
	         IF ( status .NE. ferr_ok ) GOTO 5000
              ENDIF
              valid = valid .AND. valid_y
	   ENDIF
	ENDIF

* create a space for the new result possibly with the edges flagged as bad
	CALL NEW_MR_W_EDGES( memory, cx_valid, dst_cx, dst_mr, status )
	IF ( status .NE. ferr_ok ) GOTO 5000
	is_mr( lo_isp ) = dst_mr

* cannot actually regrid if the ranges aren't valid -> all bad result
        IF ( .NOT.valid ) GOTO 1000      ! 3/93

* diagnostic message ?
	IF (mode_diagnostic) THEN
	   IF (done_by_strides) THEN
	      CALL DIAGNOSTIC_OUT('nulrgd ',dst_mr,  point_to_mr)
	   ELSE
	      CALL DIAGNOSTIC_OUT('regrid ',dst_mr,  point_to_mr)
	   ENDIF
	ENDIF

* set up 4D array limits as unsubscripted variables in COMMON
	mrlist(1) = src_mr
	mrlist(2) = dst_mr
	CALL NON_ARRAY_SUBSC( mrlist, 2 )

* reserve working array space
	IF ( xy_ave ) THEN
	   CALL GET_WORK_SPC( wk_size+wk_size2, blk1, status )
	ELSE
	   CALL GET_WORK_SPC( 2*wk_size, blk1, status )
	ENDIF
	IF ( status .NE. ferr_ok ) GOTO 5000

* do the regridding
	IF ( done_by_strides ) THEN
	   CALL COPY_GRID( memory(1,mr_blk1(src_mr)),  src_mr,
     .			   memory(1,mr_blk1(dst_mr)),  dst_mr )
	ELSEIF ( rgrd_trans .EQ. prgrd_linear_interp ) THEN
	   IF ( itsa_string ) GOTO 5010
	   CALL DO_LIN_REGRID( cx_valid, axis,
     .			memory(1,mr_blk1(src_mr)), src_mr,
     .			memory(1,mr_blk1(dst_mr)), dst_mr,
     .			memory(1, blk1), wk_size+1, 2*wk_size )
	ELSEIF ( rgrd_trans .EQ. prgrd_associate ) THEN
	   CALL DO_ASN_REGRID( cx_valid, axis,
     .			memory(1,mr_blk1(src_mr)), src_mr,
     .			memory(1,mr_blk1(dst_mr)), dst_mr )
	ELSEIF ( rgrd_trans .EQ. prgrd_average
     .	 .OR.    rgrd_trans .EQ. prgrd_variance
     .	 .OR.    rgrd_trans .EQ. prgrd_sum
     .	 .OR.    rgrd_trans .EQ. prgrd_good_pt ) THEN
	   IF ( itsa_string ) GOTO 5010
	   CALL DO_AVE_REGRID( cx_valid, axis, rgrd_trans,
     .			memory(1,mr_blk1(src_mr)), src_mr,
     .			memory(1,mr_blk1(dst_mr)), dst_mr,
     .			memory(1, blk1), wk_size+1, 2*wk_size )
	ELSEIF ( rgrd_trans .EQ. prgrd_xy_ave ) THEN
	   IF ( itsa_string ) GOTO 5010
	   CALL DO_XYAVE_REGRID( cx_valid,
     .			memory(1,mr_blk1(src_mr)), src_mr,
     .			memory(1,mr_blk1(dst_mr)), dst_mr,
     .			memory(1, blk1), wk_size+1, wk_size+wk_size2 )
	ELSEIF ( rgrd_trans.EQ.prgrd_min
     .	   .OR. rgrd_trans.EQ.prgrd_max ) THEN
	   IF ( itsa_string ) GOTO 5010
	   CALL DO_WINDOW_REGRID( cx_valid, axis, rgrd_trans,
     .			memory(1,mr_blk1(src_mr)), src_mr,
     .			memory(1,mr_blk1(dst_mr)), dst_mr,
     .			memory(1, blk1), wk_size+1, 2*wk_size )
	ELSEIF ( rgrd_trans .EQ. prgrd_xact_pts ) THEN
	   CALL DO_XACT_REGRID( cx_valid, axis,
     .			memory(1,mr_blk1(src_mr)), src_mr,
     .			memory(1,mr_blk1(dst_mr)), dst_mr,
     .			memory(1, blk1), wk_size+1, 2*wk_size )
	ELSEIF ( rgrd_trans .EQ. prgrd_nearest ) THEN
	   CALL DO_NEAREST_REGRID( cx_valid, axis,
     .			memory(1,mr_blk1(src_mr)), src_mr,
     .			memory(1,mr_blk1(dst_mr)), dst_mr,
     .			memory(1, blk1), wk_size+1, 2*wk_size )
* modulo regridding
	ELSE
	   IF ( itsa_string ) GOTO 5010
* ... for all modulo regridding cases allocate a temp variable to contain
* ... the modngd (number of valid points) result -- this is used in all
* ... of the modulo calculations.
* ... For modvar (variance) also allocate a temp space for the average
* ... which is needed in the process of computing the variance
	   ave_mr = dummy_mr	! component 3 default
!    ... (actually, CREATE_TEMP_MEM_VAR would be faster and simpler ...)
	   CALL NEW_MR_W_EDGES(memory,cx_valid,dst_cx,ngd_mr,status)
	   IF ( status .NE. ferr_ok ) GOTO 5200
	   mr_regrid_trans ( axis, ngd_mr ) =  prgrd_modngd
	   IF ( rgrd_trans .EQ. prgrd_modvar ) THEN
	      CALL NEW_MR_W_EDGES(memory,cx_valid,dst_cx,ave_mr,status)
	      IF ( status .NE. ferr_ok ) GOTO 5100
!	      mr_regrid_trans ( axis, ave_mr ) = prgrd_modulo	! bookkeeping
	   ENDIF

* set up 4D array limits and do the regridding
	   mrlist(3) = ngd_mr
	   mrlist(4) = ave_mr
	   CALL NON_ARRAY_SUBSC( mrlist, 4 )
	   CALL MODULO_REGRID( cx_valid, axis, rgrd_trans,
     .			memory(1,mr_blk1(src_mr)), src_mr,
     .			memory(1,mr_blk1(dst_mr)), dst_mr,
     .			memory(1,mr_blk1(ngd_mr)), ngd_mr,
     .			memory(1,mr_blk1(ave_mr)), ave_mr )

* ... delete temporary variables
	   CALL DELETE_VARIABLE( ngd_mr )
	   IF ( rgrd_trans .EQ.  prgrd_modvar ) THEN
	      CALL DELETE_VARIABLE( ave_mr )
	   ENDIF

	ENDIF
	CALL RELEASE_WORK_SPC

* indicate we are done with the component
 1000   CALL MR_NOT_IN_USE( src_mr )
	IF (done_by_strides) CALL DELETE_VARIABLE( src_mr ) ! invalid lims in mr

* intermediate grid ? deallocate grid space and delete intermediate data
	IF ( is_phase( lo_isp ) .EQ. prgrd_from_ntrmed ) THEN
	   CALL DEALLO_GRID( status )
	   IF ( status .NE. ferr_ok ) GOTO 5000
	   IF ( mr_protected(src_mr) .EQ. mr_not_protected )
     .			CALL DELETE_VARIABLE ( src_mr )
	ENDIF

* pop off context used by source grid
        cx_stack_ptr = cx_stack_ptr - 1
	RETURN 1

* error exits
 5000	RETURN
 5010	CALL ERRMSG( ferr_invalid_command, status,
     .		'cant use this regrid transform on strings: @'
     .		//alg_regrid_code(rgrd_trans), *5000)
 5100	CALL DELETE_VARIABLE( dst_mr )
 5200	CALL DELETE_VARIABLE( ngd_mr )
	GOTO 5000
	END	





