	SUBROUTINE IS_STRIP( act_class, idim, *, status )

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
* strip away the requirement for a transformation of a variable on the
* interpretation stack by dedicating a stack level to the evaluation of that
* transformation and putting a request for the first required component of the
* calculation onto the stack.

* programmer - steve hankin
* NOAA/PMEL, Seattle, WA - Tropical Modeling and Analysis Program
* written for VAX computer under VMS operating system
*
* revision 0.0 - 4/21/87
* ... (many revisions) ... revision 1.0 - 7/25/88 - added re-gridding code
* V200:  7/20/89 - 4D symmetrical - major re-write
*	10/16/89 - correct bug in regrid code when axis=0
*	 2/19/90 - corrected bug on RETURN when no regrid to do
*	 2/27/90 - pass through unspecified regions to be handled later
* V301:  2/24/94 - small change due to mode_arg as a 2D variable
* V320:	 12/8/94 - add modulo operations on all file variables
*	 2/21/95 - add support for reversed coordinate axes
* V420:  9/97/95 - bug fix for index=0:npts ("GT" should be "GE")
* 	   11/95 - strip of modulo wraps and allow gathering on modulo regrid
*		 - also, only gather a calculation if mode despeerate is SET
*		 - changed "csize" into "cxsize" to psych out a weird OSF bug
*	 	 - added isact_class_reconcile for command qualifiers
*		 - check for explicit limits with @ITP 
* V430:     7/96 - incorporate 4D transformations:  AVE, VAR, DIN, NDG, NBD
* V490:  9/97 - *sh* OFFSET_SS returns status
*	     - *kob* replaced call to ISUBSCRIPT w/ call to ISUBSCR_CX in order
*               to have access to context information - needed for negative
*               time step processing
* V510: *sh* 9/99 - remove warning about too-negative modulo. Change in
*	TM_WORLD has hopefully made this a legitimate operation
*	*sh* 1/00 - bug fix: @ITP failed to set subscripts if exact grid point
* V533: *sh* 6/01 - support for combined netCDF strides (cx_delta) and modulo
* V541: *sh* 2/02 - support for modulo axis lengtha
* 560: *acm* 4/04 - stride/modulo fixes: use NINT to convert REAL*8 to integer
* V602  8/06 *jli* chages for native strides 
*       *acm* 3/12 Add E and F dimensions (use nferdims in tmap_dims.parm)
* V680 5/12 *acm* 6D Ferret, changes for DEFINE DATA/AGGREGATE
* V690+ 11/13 *sh* support for auxiliary var regridding: temp[gz(depth)=zax]
* V698 1/16 *sh* fix to bug #2347 (agg res var no instead of com var on stack)
*		 alter agg setup to detect LET/D vars at eval time
*      4/16 *sh* Allow that SET VAR/NAME= has altered the agg varname
* V7   5/12 *acm* Union aggregations: DEFINE DATA/AGG/U
* V702 11/16 *sh* Allow split/gather logic to be used on axis of a compressing
*                 transform.
* V720  5/17 *sh* support for DSG operations
* V741 *acm* 5/18 Promoting global attributes to a variable in ensembles
* V741 *acm* 6/18  Last bit of 1868: multi-D trans, if requested w/ some 
*                  directions orthogonal to grid, those should be ignored
* V751 *acm* 6/19 Remove unused declarations
* V76  *acm* 3/20 Implementing regridding to point-type DSG data
* V761 *acm* 7/20 Issue 1981:in ISUBSCR_CX distinguish rounding for single 
*                 point vs range of points

        IMPLICIT NONE
	include 'tmap_dims.parm'
#   include "tmap_dset.parm"
	include	'ferret.parm'
	include	'errmsg.parm'
	include	'interp_stack.parm'
	include	'xcontext.cmn'
	include	'xvariables.cmn'
	include	'xprog_state.cmn'
	include	'xalgebra.cmn'
	include	'xax_speed.cmn'
        include 'xtm_grid.cmn_text'
        include 'xdset_info.cmn_text'   ! DSG stuff
        include 'xfr_grid.cmn'

* calling argument declarations:
	INTEGER	act_class, idim, status
* RETURN 1 - goes back to strip off any remaining transformations

* internal variable declarations:
	LOGICAL	ITSA_MODULO_REGRID, TM_ITS_SUBSPAN_MODULO,
     .		ITSA_MODULO_AXIS, TM_SAME_GRID_DEF,
     .		SCALAR_GRID_LINE, TM_ITSA_DSG, ITSA_REGRID_FROM_DSG,
     .		itsa_uvar, modulo_dest_regrid, allatonce, attoutflag, 
     .		got_it, its_dsg
	INTEGER	CGRID_SIZE_MAX, CAXIS_LEN, CAXIS_MODLEN, CX_DIM_LEN, 
     .		AUX_AXIS_FROM_ARG, CGRID_AXIS, POTENT_SIZE, 
     .		ISUBSCR_CX, TM_AXES_REVERSED, CGRID_SIZE, MODULO_PARITY,
     .		STR_SAME, TM_LENSTR1, KNOWN_GRID, 
     .		TM_DSG_DSET_FROM_GRID, TM_DSG_OBS_GRID, TM_DSG_ORIENTATION,
     .		nrev, reversed(nferdims), idel,
     .		trans, grid, dset, lo_off, hi_off, aclass,
     .		lo_cx, hi_cx, big_cx, lo_isp, i, axis,
     .		var, ss, lo_ss, hi_ss, res_size,
     .		nmod, nmodhi, npts, xlate, dup_cx, i4d,
     .          line, imemb, imemb_dset, grids_dset, len, ncomp,
     .		orig_grid, iline, attlen, maxlen, orig_cat, orig_var,
     .		memb_dset, hi_cat, hi_var, hi_grid, agg_vlen, memb_vlen, slen,
     .		orientation, itrns, ndim
	INTEGER*8 available_mem, biggest_allowed
	REAL*8	TM_WORLD, ww_given, ww_nrst
	CHARACTER*128 VAR_CODE, agg_vname, memb_vname
	CHARACTER*128 buff
	CHARACTER LEFINT*6

* --- end of introductory code ---
* initiailize - ! temp:always called at bottom of stack frame for transform
	aclass = act_class
	lo_isp = isp
	lo_cx  = is_cx( lo_isp )
	grid   = cx_grid( lo_cx )
	dset   = cx_data_set( lo_cx )
	var    = cx_variable( lo_cx )
	axis   = idim
	is_phase( lo_isp ) = unspecified_int4	! diag housekeeping
	is_sum  ( lo_isp ) = unspecified_int4	! diag housekeeping
	itsa_uvar = cx_category(lo_cx) .EQ. cat_user_var
	modulo_dest_regrid = .FALSE.		! if regrid requires modulo
	big_cx = lo_cx   ! initialize default. Used in compressed split/gather
	trans  = trans_no_transform
	its_dsg = TM_ITSA_DSG(grid)

* set up a context buffer to be modified for various grids needed
	CALL CREATE_NEW_CONTEXT( lo_cx, hi_cx, status )	! pushes stack
	IF ( status .NE. ferr_ok ) GOTO 5000

****************

 100	IF (aclass .EQ. isact_class_aggregate_gather) THEN

* SET UP TO GATHER THE COMPONENTS OF AN AGGREGATE VARIABLE
* ***==> SEE DOCUMENTATION of agg handling in header of INIT_EF_AGGREGATE_DSET
	   is_act( lo_isp ) = isact_aggregate_gather

* set result pointer to indicate no memory result has been created yet
	   is_mr( lo_isp ) = unspecified_int4

* Set up context (hi_cx) to get the first requested aggregate member
* Start with bookkeeping to prepare to request component
	   is_sum( lo_isp ) = cx_hi_ss(lo_cx, idim)
	   is_phase( lo_isp ) = cx_lo_ss(lo_cx, idim)
	   CALL TRANSFER_CONTEXT( lo_cx, hi_cx )

	   imemb = is_phase( lo_isp )
* ... there is no E-axis in an E-agg of FMRCs on the 2dtime var TF_TIMES 
	   IF (imemb .EQ. unspecified_int4) imemb = 1  ! presume TF_TIMES E-agg
	   agg_vname = VAR_CODE (cat_file_var, cx_variable(hi_cx))
	   agg_vlen = TM_LENSTR1(agg_vname)
	   CALL CD_GET_AGG_VAR_INFO (dset, agg_vname, imemb, orig_cat, 
     .                memb_dset, orig_grid, iline, orig_var, status)

* Is it the 2D Time variable for a Forecast-aggregation?  If so then
* the field was cached and retrieved by FIND_MEM_VAR_INCL_C_CACHE
	   IF (memb_dset .EQ. pagg_promotion_indicator_dset)
     .		CALL ERRMSG(ferr_internal, status,
     .		'aggregation cache failure in is_strip', *5000)

* for special case (hack) of an E-aggregation of FMRCs, the 2dtime array comes
* from one particular member
	   IF (orig_cat .EQ. cat_ef_time2d) THEN
	      hi_cat = cat_file_var
	      hi_var = orig_var
	      hi_grid = orig_grid

	   ELSEIF (orig_cat .EQ. cat_agg_gpromote) THEN
	      hi_cat = cat_agg_gpromote
	      hi_var = orig_var 
	      hi_grid = mgrid_abstract
	   ELSE

* a couple of 'delayed mode' use cases to beware -- actions the user may have
* taken **SINCE** the aggregation was defined:
*  1) user may have issued a SET VAR/NAME= (rename) of the aggregated variable
*  2) user may have issued a LET/D on a member variable
* If 1, then agg_vname will differ from memb_vname 
* If 2, then cat# and var# associated with member name may have changed
* Do a fresh "find" for the original member variable name
	      memb_vname = VAR_CODE(orig_cat, orig_var)
	      memb_vlen = TM_LENSTR1(memb_vname)
	      CALL FIND_VAR_NAME (memb_dset,memb_vname, hi_cat, hi_var )
	      IF (hi_var .EQ. munknown_var_name) CALL ERRMSG(
     .		  ferr_aggregate_error, status, 'variable '//
     .		  memb_vname(:memb_vlen)//' no longer found in agg member '//
     .		  LEFINT(imemb,slen), *5000)

* confirm grid has not changed since stored during DEFINE DATA/AGG
	      hi_grid = KNOWN_GRID(memb_dset, hi_cat, hi_var)
	      IF (hi_grid .EQ. unspecified_int4) THEN   ! can only be a uvar
* ... GET_UVAR_GRID uses the interp_stack, so bump isp
	         isp = lo_isp + 1
	         CALL GET_UVAR_GRID( hi_var, memb_dset, status )
	         isp = lo_isp  ! restore to value at start of routine
                 IF ( status .NE. ferr_ok ) GOTO 5000
	         hi_grid = KNOWN_GRID(memb_dset, hi_cat, hi_var)
	      ENDIF

* check that component grid is conformable with aggregation
* (grid may have changed due to new LET/D definition)
	      CALL CONFORMABLE_AGG_GRID(idim, memb_vname, imemb,
     .                                  grid, cat_file_var, var,
     .                                  hi_grid, hi_cat, hi_var,
     .                                  orig_grid, status )
	      IF (status .NE. ferr_ok) GOTO 5000

* scalar axes of components need not match the agg result
	      DO i = 1, nferdims
	         IF ( i .NE. idim
     .	        .AND. SCALAR_GRID_LINE(i, hi_grid, hi_cat, hi_var))
     .						CALL DEL_CX_DIM(idim, hi_cx) 
	      ENDDO

* the T axis of an F-aggregation has a single set of indices but the time
* values differ for each axis.  Indices are inherited from the lo_cx
	      IF (idim .EQ. f_dim) THEN
	         cx_by_ss(t_dim, hi_cx) = .TRUE.
	         CALL FLESH_OUT_AXIS(t_dim, hi_cx, status )
	         IF (status .NE. ferr_ok) GOTO 5000
	      ENDIF
	   ENDIF

* Set limits on agg-direction to be unspecified
	   CALL DEL_CX_DIM(idim, hi_cx)
	   cx_by_ss(idim,hi_cx) = .TRUE.

* set up the member variable
	   cx_data_set(hi_cx) = memb_dset
	   cx_category(hi_cx) = hi_cat
	   cx_variable(hi_cx) = hi_var
	   cx_grid    (hi_cx) = hi_grid
	   CALL GET_CONTEXT_BAD_FLAG(hi_cx)

	   is_uvar(lo_isp) = hi_var ! relevant if component 1 is LET/D

*-*-*-*-*-*-*-*-*-*-

	ELSEIF (aclass .EQ. isact_class_union_member) THEN

* SET UP TO RETURN A VARIABLE IN A UNION AGGREGATION
	   is_act( lo_isp ) = isact_get_union_member

* set result pointer to indicate no memory result has been created yet
	   is_mr( lo_isp ) = unspecified_int4
	   is_sum( lo_isp ) = 1  ! or something... Set these equal.
	   is_phase( lo_isp ) = 1

* Set up context (hi_cx) to get the requested aggregate member
* Start with bookkeeping to prepare to request component
	   CALL TRANSFER_CONTEXT( lo_cx, hi_cx )

	   agg_vname = VAR_CODE (cat_file_var, cx_variable(hi_cx))
	   agg_vlen = TM_LENSTR1(agg_vname)

* Get the member dataset, and the grid. imemb is the variable # in the union.

	   CALL CD_GET_VAR_ID (dset, agg_vname, imemb, status)
	   CALL CD_GET_AGG_VAR_INFO (dset, agg_vname, imemb, orig_cat, 
     .                memb_dset, orig_grid, iline, orig_var, status)

* a couple of 'delayed mode' use cases to beware -- actions the user may have
* taken **SINCE** the aggregation was defined:
*  1) user may have issued a SET VAR/NAME= (rename) of the aggregated variable
*  2) user may have issued a LET/D on a member variable
* If 1, then agg_vname will differ from memb_vname 
* If 2, then cat# and var# associated with member name may have changed
* Do a fresh "find" for the original member variable name

	      memb_vname = VAR_CODE(orig_cat, orig_var)
	      memb_vlen = TM_LENSTR1(memb_vname)
	      CALL FIND_VAR_NAME (memb_dset,memb_vname, hi_cat, hi_var )
	      IF (hi_var .EQ. munknown_var_name) CALL ERRMSG(
     .		  ferr_aggregate_error, status, 'variable '//
     .		  memb_vname(:memb_vlen)//' no longer found in agg member '//
     .		  LEFINT(imemb,slen), *5000)

* confirm grid has not changed since stored during DEFINE DATA/AGG
	      hi_grid = KNOWN_GRID(memb_dset, hi_cat, hi_var)
	      IF (hi_grid .EQ. unspecified_int4) THEN   ! can only be a uvar
* ... GET_UVAR_GRID uses the interp_stack, so bump isp
	         isp = lo_isp + 1
	         CALL GET_UVAR_GRID( hi_var, memb_dset, status )
	         isp = lo_isp  ! restore to value at start of routine
                 IF ( status .NE. ferr_ok ) GOTO 5000
	         hi_grid = KNOWN_GRID(memb_dset, hi_cat, hi_var)
	      ENDIF

* set up the member variable
	   cx_data_set(hi_cx) = memb_dset
	   cx_category(hi_cx) = hi_cat
	   cx_variable(hi_cx) = hi_var
	   cx_grid    (hi_cx) = hi_grid
	   CALL GET_CONTEXT_BAD_FLAG(hi_cx)

c	   CALL TRANSFER_CONTEXT( hi_cx, lo_cx )

	   is_uvar(lo_isp) = hi_var ! relevant if component 1 is LET/D

*-*-*-*-*-*-*-*-*-*-

**************
	ELSEIF ( its_dsg .AND. aclass.EQ.isact_class_trans) THEN

* SET UP FOR A DSG TRANSFORMATION
	   trans  = cx_trans( axis, lo_cx )
	   is_act  ( lo_isp ) = isact_trans_code + trans
	   is_phase( lo_isp ) = isphase_allatonce
	   cx_trans    ( axis, hi_cx ) = trans_no_transform
	   cx_trans_arg( axis, hi_cx ) = bad_val4
* if a compressing transformation on the orientation axis 
* (e.g. T=@AVE on a tSeries) then the source data (hi_cx) will be on the
* DSG observation grid, whereas the result (lo_cx) is on the instance grid
* see get_context_grid
	   IF ( trans .GT. trans_compress_code ) THEN
	      dset = TM_DSG_DSET_FROM_GRID(grid)
	      orientation = dsg_orientation(dset)
	      IF (axis .EQ. orientation) THEN
* ... replace obs grid with instance grid
*     FLESH_OUT_AXIS will insert the correct ss limits
	         cx_grid(hi_cx) = TM_DSG_OBS_GRID(dset)
	      ENDIF
* ... set ss limits to full nfeature or nobs as the case may be
	      CALL FLESH_OUT_AXIS( axis, hi_cx, status )
	   IF ( status .NE. ferr_ok ) GOTO 5000
	   ENDIF
* no fleshing out of axes is needed -- always get full-dset limits and always
* carry along the WW limits to use as constraints when working on the data

**************
	ELSEIF ( aclass .EQ. isact_class_trans) THEN

* SET UP FOR A TRANSFORMATION
	   trans  = cx_trans( axis, lo_cx )
	   is_act  ( lo_isp ) = isact_trans_code + trans
	   is_phase( lo_isp ) = isphase_allatonce
	   cx_trans    ( axis, hi_cx ) = trans_no_transform
	   cx_trans_arg( axis, hi_cx ) = bad_val4

* 6/98: If  the transform was marked as a 4D transform, but the grid 
*       doesn't have more than one of the requested axes, reset to 
*       the 1D transform.

	   transloop: DO itrns = 1, num_4d_trans
	     IF (trans .EQ. alg_4d_trans(itrns)) THEN 
	        ndim = 0
	        DO i = 1, nferdims
	           IF (cx_trans(i,lo_cx) .NE. trans_no_transform) ndim = ndim + 1
	        ENDDO
	
	        IF (ndim .EQ. 1) THEN
                    DO i = 1, nferdims       
                       IF ( cx_trans(i,lo_cx) .EQ. trans ) 
     .	                     cx_trans( i,lo_cx) = alg_1d_to_4d(itrns)
                    ENDDO
		   trans = alg_1d_to_4d(itrns)
		   is_act  ( lo_isp ) = isact_trans_code + trans
	        ENDIF
	        EXIT transloop
	     ENDIF
	   ENDDO transloop


* pass unspecified regions on through
	   IF ( cx_lo_ss(hi_cx,idim) .EQ. unspecified_int4
     .	 .AND.  trans .NE. trans_interpolate ) GOTO 200   ! 2/90

* possibly need different limits for component than for result
	   IF ( grid .NE. unspecified_int4 ) THEN
	      IF ( trans .LT. trans_compress_code ) THEN
	         CALL OFFSET_SS( axis, lo_cx, lo_off, hi_off, status )
		 IF ( status .NE. ferr_ok ) GOTO 5000
	         cx_lo_ss(hi_cx,axis) = cx_lo_ss(hi_cx,axis) + lo_off
	         cx_hi_ss(hi_cx,axis) = cx_hi_ss(hi_cx,axis) + hi_off

* Ticket 2301: If the requested region is at the end of an axis, so the offset
* takes the high subscript below the axis start or the low subscript above the
* axis end, definition, move the index limit(s) to the axis end. 
* (similar change in is_trans)
		 
		 IF  ( .NOT. ITSA_MODULO_AXIS(hi_cx, idim) ) THEN
		    IF (cx_hi_ss(hi_cx,axis) .LT.1) cx_hi_ss(hi_cx,axis) = 1
		    IF (cx_lo_ss(hi_cx,axis) .GT. CAXIS_LEN(axis,hi_cx)) 
     .                cx_lo_ss(hi_cx,axis) = CAXIS_LEN(axis,hi_cx)
		 ENDIF

	         cx_by_ss(axis,hi_cx) = .TRUE.
	         CALL CONFINE_AXIS( axis, hi_cx, status )
	         IF ( status .NE. ferr_ok ) GOTO 5000
* set up for interpolation
	      ELSEIF ( trans .EQ. trans_interpolate ) THEN
* ... 11/95: allow explicit "@ITP" but only with explicit coord cuz
*	when mode interp is cancelled coords get rounded early on
	         IF ( .NOT.mode_interpolate
     .		 .AND..NOT.cx_given(axis,lo_cx) ) CALL ERRMSG
     .		( ferr_invalid_command, status, 
     .                  '@ITP must be accompanied by explicit coord'
     .			//pCR
     .			//'e.g. "Z=@57@ITP" is OK but "Z=@ITP" is not',
     .								*5000 )
	         ww_given = cx_lo_ww(axis,hi_cx)
	         ss = ISUBSCR_CX(ww_given,grid,axis,hi_cx,round_sngl)
	         ww_nrst = TM_WORLD( ss, grid, axis, box_middle )
	         CALL GRID_SUBSCRIPT_EXTREMES( lo_ss, hi_ss, grid, axis )
	         cx_by_ss(axis,hi_cx) = .TRUE.
	         IF ( ww_given .LT. ww_nrst ) THEN
	            cx_lo_ss(hi_cx,axis) = MAX( ss-1, lo_ss )
	            cx_hi_ss(hi_cx,axis) = ss
	         ELSEIF( ww_given .GT. ww_nrst ) THEN
	            cx_lo_ss(hi_cx,axis) = ss
	            cx_hi_ss(hi_cx,axis) = MIN( ss+1, hi_ss )
	         ELSE
	            cx_lo_ss(hi_cx,axis) = ss
	            cx_hi_ss(hi_cx,axis) = ss
	         ENDIF
	      ENDIF
	   ENDIF

* rebuild the axis now that we've messed with it
 200	   CALL FLESH_OUT_AXIS( axis, hi_cx, status )
	   IF ( status .NE. ferr_ok ) GOTO 5000

* do the same for other transformed axes if this is a 4D transformation
	   DO 220 i4d = 1, num_4d_trans
	      IF ( trans .EQ. alg_4d_trans(i4d) ) THEN
	         DO 210 i = 1, nferdims 
	            IF (  cx_trans( i, hi_cx ) .NE. trans ) GOTO 210
	            IF ( i .EQ. idim ) GOTO 210  	! already done
	            cx_trans	( i, hi_cx ) = trans_no_transform
	            cx_trans_arg( i, hi_cx ) = bad_val4
	            CALL FLESH_OUT_AXIS( i, hi_cx, status )
	            IF ( status .NE. ferr_ok ) GOTO 5000
 210	         CONTINUE
	      ENDIF
 220	   CONTINUE

	ELSEIF( aclass .EQ. isact_class_regrid ) THEN
* SET UP FOR REGRIDDING
	   IF (its_dsg) THEN
	      CALL REGRID_TO_DSG_LIMS( lo_cx, hi_cx, status )
	      IF ( status .NE. ferr_ok ) GOTO 5000
	      axis = TM_DSG_ORIENTATION(grid)   ! really any old value ...
	      IF (axis.EQ.pfeatureType_Point) axis = pfeatureType_Trajectory
	      is_act ( lo_isp ) = isact_regrid
	   ELSEIF (ITSA_REGRID_FROM_DSG(lo_cx)) THEN
	      CALL REGRID_FROM_DSG_LIMS( lo_cx, hi_cx, status )
	      IF ( status .NE. ferr_ok ) GOTO 5000
	      axis = TM_DSG_ORIENTATION(grid)   ! really any old value ...
	      is_act ( lo_isp ) = isact_regrid
	   ELSEIF (cx_naux(lo_cx) .EQ. 0 ) THEN
* ... normal regridding
	      CALL REGRID_LIMS( lo_cx, hi_cx, axis, status, *6000 ) !ab ret->mod
	      IF ( status .NE. ferr_ok ) GOTO 5000
* ... see if no regridding was required after all
	      IF ( .NOT.cx_unstand_grid(lo_cx) ) THEN
	         cx_stack_ptr = cx_stack_ptr - 1	
	         RETURN 1					! mod 2/90
	      ENDIF
	      is_act ( lo_isp ) = isact_regrid
	   ELSE
* ... auxiliary variable regridding - set up to get the first aux var
* TEMPORARY:  ONLY HAS LOGIC TO SUPPORT 1D AT THIS POINT.  Hard-coded "1"'s
	      CALL GET_AUX_VAR_CONTEXT( lo_cx, hi_cx, 1, status )
	      IF ( status .NE. ferr_ok ) GOTO 5000
* ! temporary diagnostic check that grids of aux vars are found in get-grid phase
	      IF ( cx_grid(hi_cx) .EQ. unspecified_int4 ) 
     .		CALL ERRMSG(ferr_internal, status, 'auxvar grid??', *5000)
	      axis = AUX_AXIS_FROM_ARG(lo_cx, 1)
	      is_phase ( lo_isp ) = isphase_get_aux_var
	      is_sum ( lo_isp ) = 1        ! get the first aux var named
	      is_act ( lo_isp ) = isact_auxvar_regrid
	   ENDIF
	   aclass = isact_class_regrid

	ELSEIF( aclass .EQ. isact_class_aux_plot_vars ) THEN
* SET UP TO GET AUXILIARY ARGUMETS FOR 3-ARG PLOTS ONTO THE STACK
* ...  set up to get the first aux var
* NOTE:  ONLY HAS LOGIC TO SUPPORT 1D AT THIS POINT.  Hard-coded "1"'s
	   CALL GET_AUX_VAR_CONTEXT( lo_cx, hi_cx, 1, status )
	   IF ( status .NE. ferr_ok ) GOTO 5000
	   axis = AUX_AXIS_FROM_ARG(lo_cx, 1)
	   is_phase ( lo_isp ) = isphase_get_aux_var
	   is_sum ( lo_isp ) = 1        ! get the first aux var named
	   is_act ( lo_isp ) = isact_aux_plot_vars  ! a no-op place holder

	ELSEIF( aclass .EQ. isact_class_modulo ) THEN    ! 12/94 addition
* SET UP TO ASSEMBLE BASE COMPONENTS ONTO A MODULO AXIS
!* Let this potentially fatal error slide through because it is often safe and
!* it is a nuisance to the user to change ... I guess (???) *sh* 
!	   IF ( cx_lo_ss(lo_cx,axis) .LE. (unspecified_int4+10) ) THEN
!	      buff = VAR_TRANS(axis, lo_cx, i )
! 	      CALL WARN('Modulo axis index is too negative: '//
!     .		buff(:i))
!	      CALL WARN('This can crash FERRET!!!'//
!     .		' --> Use higher coordinate values.')
!	   ENDIF
* shift the requested range into the base modulo (1:n) limits
	   npts	   = CAXIS_MODLEN( axis, hi_cx )
	   nmod = ( cx_lo_ss(lo_cx, axis) - 1 ) / npts
	   IF ( cx_lo_ss(lo_cx, axis) .LE. 0 ) nmod = nmod - 1
	   xlate = npts * nmod
	   cx_lo_ss(hi_cx, axis) = cx_lo_ss(lo_cx, axis) - xlate
	   cx_hi_ss(hi_cx, axis) = cx_hi_ss(lo_cx, axis) - xlate
* for subspan modulo axis do not request the fictitious "n+1" point
	   IF ( TM_ITS_SUBSPAN_MODULO(CGRID_AXIS(axis, lo_cx)) ) THEN
	     IF (cx_lo_ss(hi_cx,axis).EQ.npts) THEN
	       IF (cx_hi_ss(hi_cx,axis).GT.npts) THEN
* ... improper translation was based upon void start point
	         nmod = nmod + 1
	         cx_lo_ss(hi_cx,axis) = 1
	         cx_hi_ss(hi_cx,axis) = cx_hi_ss(hi_cx,axis) - npts
	       ELSE
* ... only a void point was requested, hack is to request the last valid point
	         cx_lo_ss(hi_cx,axis) = npts - 1
	         cx_hi_ss(hi_cx,axis) = npts - 1
	       ENDIF
	     ENDIF
	     IF (cx_hi_ss(hi_cx,axis) .GE. npts)
     .	        cx_hi_ss(hi_cx,axis) = npts - 1
	   ELSE
* trim the (first) request to the limits of the data
	     IF (cx_hi_ss(hi_cx,axis) .GT. npts)
     .	        cx_hi_ss(hi_cx,axis) = npts
	   ENDIF
	   IF (cx_lo_ss(hi_cx,axis) .GT. cx_hi_ss(hi_cx,axis))
     .	     cx_lo_ss(hi_cx,axis) = cx_hi_ss(hi_cx,axis)

* if requested span exceeds axis length we can modulo from one request
* note that with strides the parity may shift in the right hand replication
*      regrid_lims only lets those cases thru to here if rqsted span le npts
	   allatonce = cx_hi_ss(lo_cx,axis)-cx_lo_ss(lo_cx,axis)
     .		  .GE. CAXIS_LEN(axis,lo_cx)
	   IF ( allatonce ) THEN
	     IF (cx_delta(axis,lo_cx) .EQ. unspecified_val8) THEN
	        cx_lo_ss(hi_cx, axis) = 1
	     ELSE
	        idel = NINT(cx_delta(axis,lo_cx))
	        cx_lo_ss(hi_cx, axis) = MODULO_PARITY(cx_lo_ss(lo_cx,axis),
     .					npts, idel)
	        allatonce = npts .EQ. (npts/idel)*idel   ! parity shift?
	     ENDIF
	   ENDIF
* ... always request full base region for modulo regridding
	   IF ( modulo_dest_regrid ) THEN
	      cx_lo_ss(hi_cx, axis) = unspecified_int4  ! implying "1"
	      cx_hi_ss(hi_cx, axis) = unspecified_int4  ! implying "npts"
!	      cx_lo_ss(hi_cx, axis) = 1
!	      cx_hi_ss(hi_cx, axis) = npts
	      cx_given(axis,hi_cx)  = .FALSE. !"explicit" lims f*ck PASS_AMBIG_
	   ENDIF
* ... rebuild the axis now that we've messed with it
	   cx_by_ss(axis,hi_cx) = .TRUE.
	   CALL FLESH_OUT_AXIS( axis, hi_cx, status )
	   IF ( status .NE. ferr_ok ) GOTO 5000
* ... clone the entire context to be requested (for sanity check in IS_MODULO)
	   CALL CREATE_NEW_CONTEXT( hi_cx,dup_cx,status ) ! pushes stack
	   IF ( status .NE. ferr_ok ) GOTO 5000
	   is_big_cx(lo_isp) = hi_cx
	   hi_cx = dup_cx		! hi_cx on top for easier sanity checks
* ... prepare to assemble the modulo pieces
* ... only a single access needed (see IS_MODULO for logic details)
	   is_act ( lo_isp ) = isact_modulo
	   IF ( cx_hi_ss(lo_cx, axis) .GE. 1 ) THEN
	      nmodhi = ( cx_hi_ss(lo_cx, axis) - 1 ) / npts
	   ELSE
	      nmodhi = cx_hi_ss(lo_cx, axis) / npts - 1
	   ENDIF
	   IF ( allatonce .OR. nmod.EQ.nmodhi 		) THEN
	      is_phase ( lo_isp ) = isphase_allatonce
	   ELSE
	      is_phase ( lo_isp ) = isphase_1
	   ENDIF
	   aclass = isact_class_modulo

	ELSEIF( aclass .EQ. isact_class_subspan_fill ) THEN
* SET UP TO INSERT PHANTOM POINT INTO VOID AT END OF SUBSPAN MODULO AXIS
* ... update the axis, omitting the top point
	   npts	= CAXIS_LEN( axis, lo_cx )
	   IF (cx_hi_ss(hi_cx, axis) .NE. npts+1) CALL ERRMSG
     .		( ferr_internal, status, 'subspan_fill_not_N+1', *5000 )
	   cx_hi_ss(hi_cx, axis) = npts
	   IF ( cx_lo_ss(hi_cx, axis) .GT. npts )
     .			     cx_lo_ss(hi_cx, axis) = npts ! was void pt, alone
	   cx_by_ss(axis,hi_cx) = .TRUE.
	   CALL FLESH_OUT_AXIS( axis, hi_cx, status )
	   IF ( status .NE. ferr_ok ) GOTO 5000
	   is_act ( lo_isp ) = isact_subspan_fill
	   aclass = isact_class_subspan_fill

	ELSEIF( aclass .EQ. isact_class_reconcile ) THEN
* SET UP TO RECONCILE POSSIBLE DIFFERENCES BETWEEN REQUEST AND WHAT IS RECEIVED
	   is_act ( lo_isp ) = isact_reconcile
	   aclass = isact_class_reconcile
	   axis = 1	! kludge: prevents crash calling ITSA_MODULO_REGRID
!			! below because FORTRAN isn't smart like C about .AND.
	ELSEIF( aclass .EQ. isact_class_reverse ) THEN    ! 2/95 addition
* The data to be read are reversed on disk -- possibly along multiple axes
* Set up here to read the data raw (reversed) under category cat_file_rev
* in its reversed ordering.  Remember that it may only be a hyperslab of
* the full data on disk so use ds_grid_start/end instead of 1 to N.
* The data values will eventually be reversed in IS_REVERSE
	   cx_category(hi_cx) = cat_file_rev
	   nrev = TM_AXES_REVERSED(var,reversed) 
	   DO 250 i = 1, nrev
	      axis = reversed(i)
	      CALL VAR_SS_LIMS( axis, lo_cx, lo_ss, hi_ss )
	      xlate = cx_lo_ss(hi_cx, axis)

!             for a strided axis
              line = grid_line(axis,grid)
              IF(line_parent(line) .NE. 0) THEN 
                  hi_ss = line_dim(grid_line(axis,grid))
                  lo_ss = 1
              ENDIF

	      cx_lo_ss(hi_cx, axis) = hi_ss + lo_ss - cx_hi_ss(hi_cx,axis) 
	      cx_hi_ss(hi_cx, axis) = hi_ss + lo_ss - xlate

* ... don't rebuild the axis: the world coords are for unreversed axis
	      cx_by_ss(axis,hi_cx) = .TRUE.
 250	   CONTINUE	         
* ... prepare to read the raw,reversed data
	   is_act ( lo_isp ) = isact_reverse
	   aclass = isact_class_reverse

	ELSE
	   WRITE (6,*) 'unsupported act class',aclass
	ENDIF

* GATHER CHECK !!!
* if the grid we are about to request (hi_cx) is going to exceed reasonable
* size limits then try to split the calculation into fragments.
* For example, TEMP[I=1:100@DIN,J=1:100@AVE,K=1:25@AVE,L=1:100@AVE]
* the required data is 100x100x25x100 - TOO BIG !!
* Similarly, modulo regrids may involve source data greatly in excess
* of the result
! DSGTBD need to support split/gather on DSG regrids - temporarily bypassed
* NOTE: because Fortran does not short-circuit IF statements, the ITSA_MODULO_REGRID
*       will always be called regardless of other values.  A check was added to
*       ITSA_MODULO_REGRID to weed out invalid calls.
	IF ( (aclass.EQ.isact_class_trans
     .		.AND. trans.GT.trans_compress_code
     .		.AND. trans.NE.trans_interpolate
     .		.AND. .NOT.its_dsg)
     . .OR.  (aclass.EQ.isact_class_regrid
     .		.AND. (its_dsg
     .		  .OR. ITSA_MODULO_REGRID(lo_cx,axis)))		) THEN
* ... create the context of the first chunk, if possible
*     SETUP_GATHER may change nothing, or it may set up split/gather
*     in hi_cx, axis and the is_* variables
	   CALL SETUP_GATHER(aclass, lo_cx, hi_cx, lo_isp, axis, status)
	   IF (status .NE. ferr_ok) GOTO 5000
	ENDIF

* diagnostic mode output: "strip --> VAR_NAME[x=lo:hi@TRN:n,D=#]"
	IF ( mode_diagnostic ) 	CALL DIAG_OP('strip', aclass, lo_cx, axis)

* push stack to go after the requested component
	is_axis ( lo_isp ) = axis
	CALL PUSH_INTERP_STACK( isp, status )
	IF ( status .NE. ferr_ok ) GOTO 5000
	is_cx( isp ) = hi_cx
	IF (  is_act (lo_isp) .EQ. isact_auxvar_regrid
     .	.AND. cx_category(hi_cx) .EQ. cat_user_var ) THEN
* ... setup to evaluate auxiliary var that is user-defined (uvar)
	   is_uvar( isp ) = cx_variable( hi_cx )
	ENDIF

* successful completion
 1000	RETURN 1

* error exit
 5000	RETURN

* possible "kludge" jump point to switch from regridding to modulo
* Note that this kludge is not actually used because adequate testing is
* now (1/9/96) done in INTERP_STACK -- still retain this documentation here
* to explain the subtleties:
* This is needed to cover a subtlety that arises when pseudo-variables are
* combined with modulo operations. For the expression "A=TEMP-TEMP+I" we would
* like "I" to run from 1 to infinity -- not to modulo wrap at Nx
* (see uvar_mod_pseudo in INTERP_STACK). For this reason modulo operations are
* deferred on LET variables that contain pseudo-variables -- the modulo op
* is performed separately on each of the components in the LET variable
* definition allowing "I" to be evaluated on the full requested range.
* The present "kludge" is to cope with LIST/I=1:BIG A[GI=@MOD]. What the user
* presumably wants is to apply the I=1:BIG to the result of the GI=@MOD. So
* in this case we will NOT defer the modulo -- we will set up the stack to do
* the modulo right now.
 6000   aclass = isact_class_modulo
	CALL WARN('Internal: unexpected modulo jump!')
	modulo_dest_regrid = .TRUE.
	GOTO 100

	END

