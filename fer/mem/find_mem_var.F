	SUBROUTINE FIND_MEM_VAR( cx, mr, by_value, valid_search )

*  This software was developed by the Thermal Modeling and Analysis
*  Project(TMAP) of the National Oceanographic and Atmospheric
*  Administration''s (NOAA) Pacific Marine Environmental Lab(PMEL),
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
* look in the list of variables in memory to see if the requested one is there
* search may be done based on subscript or on world coordinate values.

* programmer - steve hankin
* NOAA/PMEL, Seattle, WA - Tropical Modeling and Analysis Program
* written for VAX computer under VMS operating system
*
* revision 0.0 - 3/31/86
* revision 0.1 - 2/25/87 - bumps priority when variable is found
* revision 1.0 - 3/23/87 - eliminated references to mvar_norm in prep. for 0,1
*			   dimensional structures.  Added contiguousity test.
*			   Syntax changes to enhance performance.
*			   Diagnostic output.
* revision 1.1 - 5/28/87 - added "STUPID" mode
* revision 1.2 - 8/19/87 - added check on cx_trans_arg
* revision 1.3 - 9/10/87 - corrected check on cx_trans_arg
* revision 1.4 - 3/5/88  - added extraction of lower dimensioned objects
* revision 2.0 - 7/21/88 - search only through the hash pile of the context
* revision 2.1 - 10/7/88 - don't use ctxt_norm to determine if exact limits
*			   check is necessary - test to see if the context
*			   truely requires a normal location
* revision 2.1 -10/12/88 - pass bad data flag on extracted variables
* revision 2.2 - 1/4/89  - check regridding transforms
* V200:  5/18/89 - 4D symmetrical - no "extracts", no priority "bumping"
*		 - ignore "given" axes in user defined variables
* V230:  1/27/92 - special checks for transformations RSUM and WEQ
*                  and for uvlim_need_lohi used by @WEQ
*        8/18/92 - @iin matches incorrectly on top limits that aren't full box
*                  e.g. TEMP[Z=0:1000@iin] is found looking for TEMP[Z=0:0@iin]
* V420:  11/95	- speed up by bypassing NORMAL axes
*	 	- for modulo axes uvar_given "given" is not a sufficient test
*				to bypass checks
* V500: 12/98 *sh* - added support for cat_const_var ("{1,3,5}") (uncached)
*        1/99 *sh* - added support for cat_string (uncached)
*	 4/99 *sh* - check mode_arg for mode_stupid
* V510: 12/99 *sh* - modulo axis to not prevent uvlin_gvn_xact hit
* V533: 6/01 *sh* - added @EVNT - check as in trans_run_sum
*	          - check that the data type matches
*		  - check that the delta value matches
* 2/12 *kms* Add E and F dimensions (use nferdims in tmap_dims.parm)
* V685 4/13-3/14 *sh* support for auxiliary variables used in regridding
* V692 7/14 *acm* ticket 2183: handle @WGT transform as for integrals
* V695+ 5/15 *sh* added valid_search argument, to support c cache hits
* V701:  7/16 *acm* Ticket 2452. Instances where the cx_*ww and mr*ww values 
*	          should be exactly the same but the cx coordinates are ever 
*	          so slightly too wide to match the request. Use TM_DFPEQ
*	          to test for approx. equal values before rejecting the match  
*	          with what is in memory.
* V720: 4/17 *sh* special behaviors finding DSG variables 
* V74+  5/18 *acm* issue 1868: gridding transforms that will grid to same axis

* (V510) How on cache hits on external functions (EFs) work:
* uvar_given is used to encapsulate information about cache reuse for EF
* results. This ensures that the information is inherited by variables that use
* variables that use EFs. The resultant grid of an EF gets its axes from its
* arguments and from impositions where the EF asserts that the result will have
* a normal, abstract, or custom-imposed axis. These determine the value
* of uvar_given as follows. If the result axis is:
*	set to normal	-> uvlim_irrelevant (no match needed)
*	set to ABSTRACT	-> uvlim_hidden (no match needed)
*	set to custom   -> uvlim_hidden (no match needed)
*	inherited	(exact match required)
*	  with limits given in the arg definitions -> uvlim_gvn_xact
*	  with limits missing from the arg definitions -> uvlim_needed_xact
* Requiring an exact match means that the cached field will be reused only
* when the lo:hi limits of the requested context exactly match the lo:hi
* limits for this axis of the cached field. The reason that an exact match
* is required is because, in general, the function may utilize information
* from any part of the incoming data to effect all of the result. (For example
* a function that multipled one argument by the maximum value along the axis
* of another.) 
* Some of the internal grid-changing functions (e.g. RESHAPE) require
* slightly different rules for setting uvar_given -- see gcf_impose_axes.F
*

        IMPLICIT NONE
        include 'tmap_dims.parm'
#include "tmap_dset.parm"
        include 'implicit.parm'
	include 'ferret.parm'
	include 'errmsg.parm'
	include 'xdset_info.cmn_text'     ! dsg_loaded_lm
	include 'xtm_grid.cmn_text'
	external xgt_grid_data
	include 'xvariables.cmn'
	include 'xcontext.cmn'
	include 'xprog_state.cmn'

* calling argument declarations:
	INTEGER cx, mr
	LOGICAL by_value, valid_search

* internal variable declarations:
	LOGICAL TM_DFPEQ, TM_ITSA_DSG, REGRID_TO_SELF,
     .          has_trans, itsa_uvar, itsa_dsg,
     .          normal(nferdims), non_modulo(nferdims), 
     .          self(nferdims), transformed
	INTEGER HASH_CX, line, lm, status,
     .		mv, defining_grid, idim, var, cat, 
     .		dset, mline, sum
        REAL*8  TM_WORLD, ww_hi_bound, c_cache_ptr

* initialize
	valid_search = .TRUE.

* in "STUPID" mode the program cannot find anything in memory
*  (4/99) if mode_arg is 1 we are are using "weak_cache", so do not skip out
	IF (mode_stupid .AND. mode_arg(pmode_stupid,1).NE.1) GOTO 5000

* if the context carries a paux_stat_needed requirement, then a cache hit should
* not be made, because there is an auxiliary variable to be loaded as well
	DO idim = 1, nferdims
	   IF (cx_aux_stat(idim, cx) .EQ. paux_stat_needed) GOTO 5000
	ENDDO

* initialize
	defining_grid = cx_grid( cx )
	var	      = cx_variable(cx)
	cat	      = cx_category(cx)
	dset	      = cx_data_set(cx)

* is there a completed context to look for ?
	IF ( defining_grid   .EQ. unspecified_int4
     .	.OR. dset	     .EQ. unspecified_int4
     .	.OR. var             .EQ. unspecified_int4 ) GOTO 5000

* cat_const_vars are never cached (see IS_CONST_VAR)
* ditto for cat_string (as-of 1/99)
	IF ( cat .EQ. cat_const_var
     .  .OR. cat .EQ. cat_string   ) GOTO 5000

* are we looking for  user defined variable ?
	itsa_uvar = cat .EQ. cat_user_var

* are we hunting for a DSG variable?
	itsa_dsg = TM_ITSA_DSG(defining_grid)

* check for untransformed DSG coordinate variable or rowSize
* they are already cached in line memory
	IF (itsa_dsg .AND. cat.EQ.cat_file_var ) THEN
	   transformed = cx_unstand_grid(cx)
	   IF (.NOT.transformed) THEN
	      DO idim = 1, nferdims
	         IF (cx_trans(idim, cx) .NE. trans_no_transform) THEN
	            transformed = .TRUE.
	            EXIT
	         ENDIF
	      ENDDO
	   ENDIF
	   lm = dsg_loaded_lm(var)
	   IF (.NOT.transformed .AND. lm.EQ.int4_init) THEN
	      sum = 0
	      IF (lm.EQ.dsg_row_size_var(dset)) sum = sum+1
	      DO idim = 1,t_dim
	          IF (dsg_coord_var(idim,dset) .EQ. int4_init) CYCLE
		  IF (lm.EQ.dsg_loaded_lm(dsg_coord_var(idim,dset))) sum = sum+1
	      ENDDO 
* ... create and return an mr table entry pointing to the dsg_linemem
	      IF (sum.GT.0) THEN
	         CALL CREATE_LM_TABLE_ONLY_MEM_VAR( cx, lm, mr, status )
                 IF (status .EQ. ferr_ok) GOTO 1000
	      ENDIF
* ... in rare chance of error status, the cache hit failed
	   ENDIF
	ENDIF

* do we need special checks for transforms ?
	has_trans = .FALSE.
	DO idim = 1, nferdims
	   has_trans = has_trans
     .        .OR. (cx_trans(idim,cx).NE.trans_no_transform)
	ENDDO

* flag the normal axes so we can bypass checks on them quickly
* DSG grids don't have ordinary modulo-ing, don't handle regions as modulo here

	DO 210 idim = 1, nferdims
	   line = grid_line(idim,defining_grid)
	   IF (line .EQ. mnormal) THEN
	      IF (itsa_dsg .AND. idim.NE.f_dim) THEN
* ... DSG constraints are relevant on some mnormal grid axes
*     e.g. time series features can be subsetted based upon X,Y position
! huh?  how can this compile?
	         normal(idim) = cx_lo_ww(idim,cx) .EQ. unspecified_val8
	      ELSE
	         normal(idim) = .TRUE.
	      ENDIF
	   ELSE 
	      normal(idim) = .FALSE.
	      non_modulo(idim) = .NOT.line_modulo(line)
	   ENDIF
	   IF (itsa_dsg) non_modulo(idim) = .FALSE.
 210	CONTINUE

* determine memory hash pile to search
	mv = HASH_CX( cx )

* locate next table slot in linked list
 100	mv = mv_flink( mv )
	IF ( mv .LE. 0 ) THEN
	   mr = mr_nonexist
	   RETURN     ! not found in normal Ferret cache
	ENDIF

* right variable ?
	IF ( mr_variable ( mv ) .NE. var
     .	.OR. mr_category ( mv ) .NE. cat
     .	.OR. mr_data_set ( mv ) .NE. dset
     .	.OR. mr_grid ( mv ) .NE. defining_grid       ) GOTO 100

* check variable type
	IF ( mr_type(mv) .NE. cx_type(cx) ) GOTO 100

* DSGTBD: the F axis of a DSG can be treated as a gridded axis: superset ok

* flag for transforms that allow regrid-to-same axis

 	DO idim = 1, nferdims
	   line = grid_line(idim,defining_grid)
	   mline = grid_line(idim,mr_grid(mv))
	   self(idim) = line .EQ. mline .AND. 
     .       REGRID_TO_SELF (mr_regrid_trans(idim, mv) )
           IF (self(idim)) GOTO 100
	ENDDO

* right instance (coordinate ranges and transformations) ?

	IF ( itsa_uvar ) THEN

* * * * * LOOK FOR A USER-DEFINED VARIABLE BASED ON COORDINATE VALUES  * * * * 
	   DO 40 idim = 1,nferdims

	      IF ( normal(idim) ) GOTO 40

* transformation correct ?
	      IF ( mr_trans(idim,mv) .NE. cx_trans(idim,cx) ) GOTO 100

* limits specified in uvar expression ? -->  no match necessary
* 11/95 - since modulo is not stripped until after FIND_MEM_VAR is called
*	  we cannot bypass this on modulo axes.
*		e.g. x1 has I=1:10; let xgap=x1[i=3:7]
*		then list xgap[i=1:15] should not "find" the i=3:7 result
* 12/99 - if uvlim_gvn_xact then modulo is never stripped in INTERP_STACK
*	(note: when is it ever stripped for uvlim_gvn_*?. This logic looks
*		out of date. Maybe the non_modulo restriction can be lifted)

	      IF ( (uvar_given(idim,var) .EQ. uvlim_gvn_xact) .OR.
     .		   ( (uvar_given(idim,var) .LT. uvlim_gvn_xact) .AND.
     .		      non_modulo(idim) )  ) GOTO 40

* insist that the  grid stored in memory encompasses the requested data
	      IF (mr_lo_ww(idim, mv) .GT. cx_lo_ww(idim, cx)) THEN
                 IF ( .NOT. TM_DFPEQ(cx_lo_ww(idim,cx),mr_lo_ww(idim,mv)) ) GOTO 100
	      ENDIF
	      IF (mr_hi_ww(idim, mv) .LT. cx_hi_ww(idim, cx)) THEN
                 IF ( .NOT. TM_DFPEQ(cx_hi_ww(idim,cx),mr_hi_ww(idim,mv)) ) GOTO 100
	      ENDIF

* insist that the delta values match
	      IF ( mr_delta(idim,mv) .NE. cx_delta(idim,cx) ) THEN
                 IF ( .NOT. TM_DFPEQ(mr_delta(idim,mv),cx_delta(idim,cx)) )  GOTO 100
	      ENDIF

 40	   CONTINUE

* right instance (coordinate ranges and transformations) ?
	ELSEIF ( by_value ) THEN

* * * * * * LOOK FOR A VARIABLE BASED ON COORDINATE VALUES * * * * * * * * 
	   DO 50 idim = 1,nferdims

	      IF ( normal(idim) ) GOTO 50

* transformation correct ?
	      IF ( mr_trans(idim,mv) .NE. cx_trans(idim,cx) ) GOTO 100

* If it's a DSG grid, then a change to the context in any direction  
* requires redoing the computation -- the context applies to the list
* of features/ obs.

	      IF (itsa_dsg) THEN 
                 IF ( .NOT. TM_DFPEQ(cx_lo_ww(idim,cx),mr_lo_ww(idim,mv)) ) GOTO 100
                 IF ( .NOT. TM_DFPEQ(cx_hi_ww(idim,cx),mr_hi_ww(idim,mv)) ) GOTO 100
	      ENDIF

* insist that the  grid stored in memory encompasses the requested data

	      IF (mr_lo_ww(idim, mv) .GT. cx_lo_ww(idim, cx)) THEN
                 IF ( .NOT. TM_DFPEQ(cx_lo_ww(idim,cx),mr_lo_ww(idim,mv)) ) GOTO 100
	      ENDIF
	      IF (mr_hi_ww(idim, mv) .LT. cx_hi_ww(idim, cx)) THEN
                 IF ( .NOT. TM_DFPEQ(cx_hi_ww(idim,cx),mr_hi_ww(idim,mv)) ) GOTO 100
	      ENDIF

* insist that the delta values match
	      IF ( mr_delta(idim,mv) .NE. cx_delta(idim,cx) ) THEN
                 IF ( .NOT. TM_DFPEQ(mr_delta(idim,mv),cx_delta(idim,cx)) ) GOTO 100
	      ENDIF

 50	   CONTINUE

	ELSE		! by subscript

* * * * * * LOOK FOR A VARIABLE BASED ON SUBSCRIPT VALUES * * * * * * * * *
	   DO 60 idim = 1,nferdims

	      IF ( normal(idim) ) GOTO 60

* transformation correct ?
	      IF ( mr_trans(idim,mv) .NE. trans_no_transform ) GOTO 100
	      IF ( cx_trans(idim,cx) .NE. trans_no_transform ) 
     .		 STOP 'find_by_ss_trans'	! temp diagnostic

* insist that the  grid stored in memory encompasses the requested data
* ... this is an adequate test because the grids looked for by subscript are
* ... always untransformed --> normal axes always have a span on 1 point
	      IF ( mr_lo_ss(mv, idim) .GT. cx_lo_ss(cx, idim)
     .	      .OR. mr_hi_ss(mv, idim) .LT. cx_hi_ss(cx, idim)) GOTO 100

* insist that the delta values match
	      IF ( mr_delta(idim,mv) .NE. cx_delta(idim,cx) ) THEN
	         IF ( .NOT. TM_DFPEQ(mr_delta(idim,mv),cx_delta(idim,cx)) ) GOTO 100
	      ENDIF

 60	   CONTINUE
	ENDIF

* make sure it is completed
	IF ( mr_protected( mv ) .EQ. mr_in_progress ) GOTO 100

* special tests left for last to keep the main loop simple and fast
* ... special limit requirements for indefinite integrals:
*     to satisfy the search an indefinite integral must have exactly the same
*     lower limit and must have upper limit either identical or the grid
*     we are looking for must have upper endpoint on a grid box boundary
*     ( to avoid errors of partial grid boxes )
* fixed logic bug: 8/92 using ww_hi_bound
** was:            IF ( mr_hi_ww(idim,mv).NE.cx_hi_ww(idim,cx)
**     .          .AND. mr_hi_ss(mv,idim).EQ.cx_hi_ss(cx,idim) ) GOTO 100
*
	IF ( has_trans ) THEN
	   DO 70 idim = 1, nferdims
	      IF ( normal(idim) ) GOTO 70
	      IF ( cx_trans( idim, cx ) .EQ. trans_integ_indef .OR. 
     .             cx_trans( idim, cx ) .EQ. trans_avg_wgt .OR. 
     .             cx_trans( idim, cx ) .EQ. trans_4d_avg_wgt) THEN
	         IF ( mr_lo_ww(idim,mv).NE.cx_lo_ww(idim,cx) ) GOTO 100
                 IF ( mr_hi_ww(idim,mv).NE.cx_hi_ww(idim,cx) ) THEN
                    ww_hi_bound = TM_WORLD( cx_hi_ss(cx,idim),
     .                         defining_grid, idim, box_hi_lim ) 
                    IF ( ww_hi_bound .NE. cx_hi_ww(idim,cx)) GOTO 100
                 ENDIF
	      ENDIF
 70	   CONTINUE
	ENDIF

	IF ( has_trans ) THEN
	   DO 80 idim = 1, nferdims
	      IF ( normal(idim) ) GOTO 80
* ... argument of transform must be right
* ... and transforms that compress to a point must have exact limits
	      IF ( cx_trans_arg(idim,cx).NE.mr_trans_arg(idim,mv) ) GOTO 100
	      IF ( cx_trans(idim,cx) .GT. trans_compress_code ) THEN
	         IF ( mr_lo_ww(idim,mv).NE.cx_lo_ww(idim,cx) ) GOTO 100
	         IF ( mr_hi_ww(idim,mv).NE.cx_hi_ww(idim,cx) ) GOTO 100
	      ENDIF
* ... @RSUM must match lower subscript exactly (1/92)
* ... @EVNT ditto (6/01)
	      IF ( cx_trans( idim, cx ) .EQ. trans_run_sum
     .        .OR. cx_trans( idim, cx ) .EQ. trans_event_mask ) THEN
                 IF ( mr_lo_ss(mv,idim).NE.cx_lo_ss(cx,idim) ) GOTO 100
              ENDIF
* ... @WEQ must match both hi and lo subscripts (1/92)
	      IF ( cx_trans( idim, cx ) .EQ. trans_wgted_eq ) THEN
                 IF ( mr_lo_ss(mv,idim).NE.cx_lo_ss(cx,idim)
     .           .OR. mr_hi_ss(mv,idim).NE.cx_hi_ss(cx,idim) ) GOTO 100
              ENDIF

 80	   CONTINUE
	ENDIF

* ... if regridded then regridding transforms must be right
	IF ( cx_unstand_grid( cx ) ) THEN
	   DO idim = 1, nferdims
	      IF ( normal(idim) ) CYCLE
	      IF ( self(idim) ) CYCLE
	      IF ( cx_regrid_trans( idim, cx ) .NE.
     .		   mr_regrid_trans( idim, mv   ) ) GOTO 100
	   ENDDO

* ... check for auxiliary regridding match
	   IF ( mr_naux( mv ) .NE. cx_naux( cx ) ) GOTO 100
	   IF ( cx_naux( cx ) .GT. 0 ) THEN
	      DO 91 idim = 1, nferdims
	         IF ( cx_aux_var(idim,cx) .NE. mr_aux_var(idim, mv)
     .	        .OR.  cx_aux_cat(idim,cx) .NE. mr_aux_cat(idim, mv) )
     .								GOTO 100 
 91	      CONTINUE
	   ENDIF
	ENDIF

* ... if user-defined then watch out for special limit match requirements
	IF ( itsa_uvar ) THEN
	   DO 92 idim = 1, nferdims
	      IF ( normal(idim) ) GOTO 92
	      IF ( uvar_given(idim,var) .EQ. uvlim_need_xact
     .        .OR. uvar_given(idim,var) .EQ. uvlim_need_lohi ) THEN
	         IF ( mr_lo_ww(idim,mv) .NE. cx_lo_ww(idim,cx)
     .           .OR. mr_hi_ww(idim,mv) .NE. cx_hi_ww(idim,cx)) GOTO 100
	      ELSEIF ( uvar_given(idim,var) .EQ. uvlim_need_lo ) THEN
	         IF ( mr_lo_ww(idim,mv) .NE. cx_lo_ww(idim,cx)
     .           .OR. mr_hi_ww(idim,mv) .LT. cx_hi_ww(idim,cx)) GOTO 100
	      ENDIF
 92	   CONTINUE
	ENDIF

* WE FOUND IT !
	mr = mv

* ... diagnostic message ?
 1000	IF (mode_diagnostic) CALL DIAGNOSTIC_OUT('found  ',mr,point_to_mr)

* ... send back pointer
	RETURN

* failure exit
 5000	mr = mr_nonexist
	valid_search = .FALSE.
	RETURN
	END
