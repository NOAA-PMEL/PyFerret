	SUBROUTINE IS_STRIP( act_class, idim, *, status )

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

* calling argument declarations:
	INTEGER	act_class, idim, status
* RETURN 1 - goes back to strip off any remaining transformations

* internal variable declarations:
	LOGICAL	ITSA_MODULO_REGRID, itsa_uvar, modulo_dest_regrid
	INTEGER	CGRID_SIZE_MAX, CAXIS_LEN, CX_DIM_LEN, POTENT_SIZE,
     .		ISUBSCR_CX, TM_AXES_REVERSED, CGRID_SIZE,
     .		nrev, reversed(4),
     .		trans, xtrans, grid, dset, lo_off, hi_off, aclass,
     .		lo_cx, hi_cx, big_cx, lo_isp, lo_mr, int_arg, i, axis,
     .		cxsize, split_ax, split_ax_len, frag, var, ss, lo_ss,
     .		hi_ss, nmod, nmodhi, npts, xlate, dup_cx, i4d

	REAL	ax_frac
	REAL*8	TM_WORLD, ww_given, ww_nrst
	CHARACTER VAR_TRANS*48, buff*48

#ifdef unix
	include 'tmap_format/tmap_dims.parm'
	include	'ferret_cmn/ferret.parm'
	include	'ferret_cmn/errmsg.parm'
	include	'ferret_cmn/interp_stack.parm'
	include	'ferret_cmn/xcontext.cmn'
	include	'ferret_cmn/xvariables.cmn'
	include	'ferret_cmn/xprog_state.cmn'
	include	'ferret_cmn/xalgebra.cmn'
	include	'ferret_cmn/xax_speed.cmn'
#else
	INCLUDE 'TMAP_FORMAT:TMAP_DIMS.PARM'
	INCLUDE	'FERRET_CMN:FERRET.PARM'
	INCLUDE	'FERRET_CMN:ERRMSG.PARM'
	INCLUDE	'FERRET_CMN:INTERP_STACK.PARM'
	INCLUDE	'FERRET_CMN:XCONTEXT.CMN'
	INCLUDE	'FERRET_CMN:XVARIABLES.CMN'
	INCLUDE	'FERRET_CMN:XPROG_STATE.CMN'
	INCLUDE	'FERRET_CMN:XALGEBRA.CMN'
	INCLUDE	'FERRET_CMN:XAX_SPEED.CMN'
#endif

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

* set up a context buffer to be modified for various grids needed
	CALL CREATE_NEW_CONTEXT( lo_cx, hi_cx, status )	! pushes stack
	IF ( status .NE. ferr_ok ) GOTO 5000

****************

 100	IF ( aclass .EQ. isact_class_trans) THEN

* SET UP FOR A TRANSFORMATION
	   trans  = cx_trans( axis, lo_cx )
	   is_act( lo_isp ) = isact_trans_code + trans
	   cx_trans    ( axis, hi_cx ) = trans_no_transform
	   cx_trans_arg( axis, hi_cx ) = bad_val4

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
	         ss = ISUBSCR_CX(ww_given,grid,axis,hi_cx,round_dn)
	         ww_nrst = TM_WORLD( ss, grid, axis, box_middle )
	         CALL GRID_SUBSCRIPT_EXTREMES( lo_ss, hi_ss, grid, axis )
	         cx_by_ss(axis,hi_cx) = .TRUE.
	         IF ( ww_given .LT. ww_nrst ) THEN
	            cx_lo_ss(hi_cx,axis) = MAX( ss-1, lo_ss )
	            cx_hi_ss(hi_cx,axis) = ss
	         ELSEIF( ww_given .GT. ww_nrst ) THEN
	            cx_lo_ss(hi_cx,axis) = ss
	            cx_hi_ss(hi_cx,axis) = MIN( ss+1, hi_ss )
	         ENDIF
	      ENDIF
	   ENDIF

* rebuild the axis now that we've messed with it
 200	   CALL FLESH_OUT_AXIS( axis, hi_cx, status )
	   IF ( status .NE. ferr_ok ) GOTO 5000

* do the same for other transformed axes if this is a 4D transformation
	   DO 220 i4d = 1, num_4d_trans
	      IF ( trans .EQ. alg_4d_trans(i4d) ) THEN
	         DO 210 i = 1, 4 
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
	   CALL REGRID_LIMS( lo_cx, hi_cx, axis, status, *6000 ) !ab ret -> mod
	   IF ( status .NE. ferr_ok ) GOTO 5000
* ... see if no regridding was required after all
	   IF ( .NOT.cx_unstand_grid(lo_cx) ) THEN
	      cx_stack_ptr = cx_stack_ptr - 1	
	      RETURN 1					! mod 2/90
	   ENDIF
	   is_act ( lo_isp ) = isact_regrid
	   aclass = isact_class_regrid

	ELSEIF( aclass .EQ. isact_class_modulo ) THEN    ! 12/94 addition
* SET UP TO ASSEMBLE BASE COMPONENTS ONTO A MODULO AXIS
* Let this potentially fatal error slide through because it is often safe and
* it is a nuisance to the user to change ... I guess (???) *sh* 
	   IF ( cx_lo_ss(lo_cx,axis) .LE. (unspecified_int4+10) ) THEN
	      buff = VAR_TRANS(axis, lo_cx, i )
 	      CALL WARN('Modulo axis index is too negative: '//
     .		buff(:i))
	      CALL WARN('This can crash FERRET!!!'//
     .		' --> Use higher coordinate values.')
	   ENDIF
	   npts	   = CAXIS_LEN( axis, lo_cx )
	   IF ( cx_lo_ss(lo_cx, axis) .GE. 1 ) THEN
	      nmod = ( cx_lo_ss(lo_cx, axis) - 1 ) / npts
	   ELSE
	      nmod = cx_lo_ss(lo_cx, axis) / npts - 1
	   ENDIF
	   xlate = npts * nmod
	   cx_lo_ss(hi_cx, axis) = cx_lo_ss(lo_cx, axis) - xlate
	   cx_hi_ss(hi_cx, axis) = MIN( npts, 
     .				   cx_hi_ss(lo_cx, axis) - xlate )
	   IF ( cx_hi_ss(lo_cx, axis)-cx_lo_ss(lo_cx,axis) .GE. npts )
     .	      cx_lo_ss(hi_cx, axis) = 1
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
	   IF ( cx_hi_ss(hi_cx,axis)-cx_lo_ss(hi_cx,axis)+1 .GE. npts
     .	   .OR. nmod .EQ. nmodhi 				) THEN
	      is_phase ( lo_isp ) = isphase_allatonce
	   ELSE
	      is_phase ( lo_isp ) = isphase_1
	   ENDIF
	   aclass = isact_class_modulo

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
* after the L and K transforms have been split off we will have
* TEMP[I=1:100@DIN,J=1:100@AVE,K=1:25,L=1:100] which is a reasonable 2500 pts.
* after the next strip, however, the request will be 100 times larger.  At this
* time we would be better to split up the context, say, along the L axis so
* that each requested component is a manageable size.
* 11/95: similarly, modulo regrids may involve source data greatly in excess
* of the result
 300	IF ( aclass.EQ.isact_class_trans
     .		.AND. trans.GT.trans_compress_code
     . .OR.  aclass.EQ.isact_class_regrid
     .		.AND. ITSA_MODULO_REGRID(lo_cx, axis)		) THEN
	   IF ( aclass.EQ.isact_class_trans ) THEN 
	      cxsize = CGRID_SIZE( hi_cx )	! well tested of old (11/95)
	   ELSE
	      cxsize = CGRID_SIZE_MAX( hi_cx )	! unknown lims get len=line_dim
	   ENDIF

* do we have a size problem ?
	   IF ( cxsize .LE. mode_arg( pmode_desparate,1 ) ) GOTO 800
* yes, too big ! can we split the calculation along some axis ?
*  ( try to split along a slow access axis of the data )
	   DO 500 i = 4, 1, -1
	      IF ( dset.GT.pdset_irrelevant .AND. dset.LE.maxdsets ) THEN
	         split_ax = ds_ax_speed(i, dset)
	      ELSE
	         split_ax = i
	      ENDIF
	      IF ( cx_trans(split_ax,lo_cx) .NE. trans_no_transform ) GOTO 500
	      IF ( ITSA_MODULO_REGRID(lo_cx,split_ax) ) GOTO 500
* ... beware user-defined vars with limits specified in definition
	      IF ( itsa_uvar ) THEN
	         IF ( uvar_given(split_ax,var) .NE. uvlim_needed ) GOTO 500
	      ENDIF
	      split_ax_len = CX_DIM_LEN(split_ax,lo_cx)
	      IF ( split_ax_len .EQ. 1 ) GOTO 500
* ... in desparation, only, consider splitting along the fastest access axis
	      IF ( i.EQ.1 .AND. .NOT.mode_desparate )  GOTO 500
	      GOTO 510
 500	   CONTINUE	   
* no way to split up the calculation
	   GOTO 800
* OK - lets divvie this baby up !
 510	   aclass = isact_class_gather
	   is_act(lo_isp) = isact_gather
	   axis = split_ax
* save the default context so it can be re-used at each chunk
	   CALL CREATE_NEW_CONTEXT( lo_cx, big_cx, status )	! pushes stack
	   IF ( status .NE. ferr_ok ) GOTO 5000
	   is_big_cx( lo_isp ) = big_cx
* cancel result pointer to indicate no result yet
	   is_mr( lo_isp ) = unspecified_int4
* determine the largest fragment satisfying the "desparate mode" argument
* for compressing transforms consider here ALL of the axes that may expand 
	   IF ( aclass.EQ.isact_class_trans ) cxsize = POTENT_SIZE(lo_cx)
	   ax_frac = FLOAT(mode_arg(pmode_desparate,1)) / cxsize
	   frag = MAX( 1, INT(split_ax_len*ax_frac) )
	   is_sum( lo_isp ) = frag
* set up context to get the first fragment
	   CALL TRANSFER_CONTEXT( big_cx, hi_cx )
	   cx_hi_ss(hi_cx,axis) = cx_lo_ss(hi_cx,axis) + frag - 1
	   cx_by_ss(axis,hi_cx) = .TRUE.
	   CALL FLESH_OUT_AXIS( axis, hi_cx, status )
	   IF ( status .NE. ferr_ok ) GOTO 5000
	ENDIF

* diagnostic mode output: "strip --> VAR_NAME[x=lo:hi@TRN:n,D=#]"
 800	IF ( mode_diagnostic ) 	CALL DIAG_OP('strip', aclass, lo_cx, axis)

* push stack to go after the requested component
	is_axis ( lo_isp ) = axis
	CALL PUSH_INTERP_STACK( isp, status )
	IF ( status .NE. ferr_ok ) GOTO 5000
	is_cx( isp ) = hi_cx

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

