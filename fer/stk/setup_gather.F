	SUBROUTINE SETUP_GATHER( aclass, res_cx, src_cx, lo_isp,
     .				  axis, status)

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
* GATHER CHECK !!!
* This routine is called, to determine if a split/gather operation is needed,
* and if so to create the context descriping the "chunks" to
* be gathered and the axis (or axes) along which gathering will occur

* If the grid we are about to request (hi_cx) is going to exceed reasonable
* size limits then try to split the calculation into fragments.
* For example, TEMP[I=1:100@DIN,J=1:100@AVE,K=1:25@AVE,L=1:100@AVE]
* the required data is 100x100x25x100 - TOO BIG !!
* after the L and K transforms have been split off we will have
* TEMP[I=1:100@DIN,J=1:100@AVE,K=1:25,L=1:100] which is a reasonable 2500 pts.
* after the next strip, however, the request will be 100 times larger.  At this
* time we would be better to split up the context, say, along the L axis so
* that each requested component is a manageable size.
* Similarly, modulo regrids may involve source data greatly in excess
* of the result

* NOTE that this routine does a precice job for a single transformation
* (including "4d" transforms) and modulo regridding. Where it is weak is
* when a single expression contains more than one transform -- e.g.
* "V[x=@ave,t=@sum]" or "V[gt=clim@mod,x=@ave]".  In general such syntaxes
* should be discouraged.
* V702  3/17 *sh* Code extracted from IS_STRIP and several refinements added
* V720 5/17 *sh* added support for DSG regridding splits
*      fixed undetected bug in POTENT_AX_LIMS call for min_mem_used below
* V7.21 9/17 *acm Fix ticket 2561. Multiple transformations and multiple-axis 
*                 transformations on the split axis.stk/setup_gath
* V740  3/18 *acm* Introduce parameter abstractax_len for abstract axis length 99999999
* V740  3/18 *acm* Fix issue 1964: unending loop on scatter/gather when grid length
*                  returned as abstract-axis length

        IMPLICIT NONE
	include 'tmap_dims.parm'
	include	'ferret.parm'
	include	'errmsg.parm'
	include	'interp_stack.parm'
	include	'xcontext.cmn'
	include	'xvariables.cmn'
	include	'xprog_state.cmn'
	include	'xalgebra.cmn'
	include	'xax_speed.cmn'
        include 'xtm_grid.cmn_text'

* calling arguments:
*  res_cx - (input) (aka lo_cx) context of the result, including
*                   the transform (e.g. @AVE)
*  src_cx - (inout) (aka hi_cx) On input the full context that must
*                   be processed in order for Ferret to compute the transform.
*                   On output it is the context of the first chunk
*  axis   - (inout) If splitting occurs it may not be along the transform axis 
 
* calling argument declarations:
	INTEGER	aclass, res_cx, src_cx, lo_isp, axis, status

* internal variable declarations:
	LOGICAL ITSA_MODULO_REGRID, TM_ITSA_DSG,
     .		itsa_uvar, split_on_compressed_ax, dup_split,
     .		inadequate_split_length, use_caution, no_axis_found,
     .		multiax_trans(nferdims),
     .		ax_splittable(nferdims),
     .		ax_compressed(nferdims),
     .		its_dsg
	INTEGER	POTENT_SIZE, CGRID_UVAR_SIZE,
     .		i, i1, i2, dset, var, cat, big_cx, split_ax, ax_trans,
     .		frag, lo_ss, hi_ss, trans, nbuffers, ax1, ax2,
     .		split_ax_len, ncomp, nsplittable,
     .		n_dfrnt_xforms, ax_len(nferdims)
	INTEGER*8 CGRID_SIZE_MAX, available_mem, set_aside_mem, free_mem,
     .		deferred_essential_mem, min_mem_used, res_size, src_size,
     .		ax_min_mem(nferdims)
	REAL	rsrc_size, reduction_factor_needed

******
* logic in this routine goes in 4 phases
* 0. Initialize: amount of memory available, the size of the result, etc.
* 1. Determine if splitting is needed              (exit if no)
* 2. Find the slowest single axis split that fits  (skip to step 4 if found)
* 3. Find the slowest 2-axis split that fits       (skip to step 4 if found)
*      If no solution so far, reduce the memory held in reserve and goto 2
* 4. Setup the split/gather
******

* initialize
	dset = cx_data_set( res_cx )
	var  = cx_variable( res_cx )
	cat  = cx_category( res_cx )
	itsa_uvar = cat .EQ. cat_user_var
	use_caution = mode_state(pmode_frugal,1)
	status = ferr_ok
	IF ( aclass.EQ.isact_class_trans ) THEN
	   its_dsg = .FALSE.
	ELSE
	   its_dsg = TM_ITSA_DSG(cx_grid(res_cx))
	ENDIF

* if we are returning here as part of a multi-axis split, then the isp stack
* level just above us will also be isact_gather.  If so, learn the amount of
* memory of memory that the previous axes have committed
	IF (lo_isp .LT. 2) RETURN ! should never happen. Playing it safe.
	IF (is_act(lo_isp-1) .EQ. isact_gather) THEN
	   deferred_essential_mem = is_deferred_mem(lo_isp-1)
	ELSE
	   deferred_essential_mem = 0
	ENDIF	

****************************************************************
* ***** PHASE 1. Determine if splitting is needed   (exit if no)

* how much memory will be needed for the full source data of the transform?
	IF ( aclass.EQ.isact_class_trans ) THEN 
!	   src_size = CGRID_SIZE( src_cx )
! ... have to use POTENT_SIZE because of cases like V[X=@SUM,T=@AVE]
	   src_size = POTENT_SIZE( res_cx )
	   trans  = cx_trans( axis, res_cx )
	ELSE					! modulo regrid
! ... FIXME: CGRID_SIZE_MAX is buggy, and a calc like POTENT_SIZE is needed
!            if modulo regridding and @AVE are combined on a single variable
!        For a regrid to DSG the source data size calc should be correct
	   src_size = CGRID_SIZE_MAX( src_cx )	! unknown lims get len=line_dim
	   trans = trans_no_transform
	ENDIF

* If it has an abstract axis, just nominally set a size.  The actual limits
* will be evaluated as the computation proceeds.

	IF (src_size .EQ. FLOAT(abstractax_len)) THEN
	   src_size = 1000
	ENDIF 

* locate other axes of a multi-axis transform (e.g. xy average)
* and determine how many *different* compressing transforms there are
	n_dfrnt_xforms = 0
        DO i = 1, nferdims
           multiax_trans(i) = cx_trans(i, res_cx) .EQ. trans
	   IF (.NOT.multiax_trans(i)
     .    .AND. (cx_trans(i,res_cx) .GT. trans_compress_code) )
     .		n_dfrnt_xforms = n_dfrnt_xforms + 1
        ENDDO

* how many result buffers are needed?
* ... to split/gather on a compressed axis requires
*     1) the result
*     2) the accumulation buffer (e.g. @ave requires a running sum)
*     3) additional accumulation buffer for @VAR
* ... modulo regrids require 3 buffers, plus one more for modvar (see IS_TRANS)
*     If split/gather occurs on the compressed axis, then the buffers are
*     part of the result size.  If the split/gather occurs on another axis
*     then the buffers are part of the chunk size (the "source" data - src)
	IF (aclass .EQ. isact_class_trans) THEN
	   IF (trans .EQ. trans_variance) THEN
	      nbuffers = 3
	   ELSE
	      nbuffers = 2
	   ENDIF
	ELSEIF (its_dsg) THEN
	   nbuffers = 2  !   really, 1.5 since DSG result usually much smaller
	ELSE
	   IF (cx_regrid_trans(axis,res_cx) .EQ. prgrd_modvar) THEN
	      nbuffers = 4   ! see IS_REGRID
	   ELSE
	      nbuffers = 3
	   ENDIF
	ENDIF

*  ...to compute a user variable often requires temporary vars and other
*     component vars to be held in memory at once as function arguments
*     As a default cautious assumption we will allow a uvar to use only
*     1/4 as much memory, figuring there may be 4 of them in memory at once
*     NOTE: This is the crudest heuristic used in this memory management (3/17)
*     It forces Ferret to use much less memory on a uvar, even if its
*     definition is simply V=fvar, which needs no extra memory at all because
*     it uses RE_ASSIGN_VARIABLE in IS_ALGEBRA.  The best improvement I have
*     been able to come up with for this would be that during get_grid phase
*     in addition to a grid we compute a "max_significant_stack_depth" for
*     each uvar.  This would count only the fvars and uvars (not constants and
*     pseudo-vars which are small) that need to be on the stack simultaneously
*     In pyramids of definitions, the stack requirements of child vars would
*     need to factor into this calculation (heuristically, cuz the exact stack
*     max cannot be determined in this simple way).
	IF (itsa_uvar) THEN
	   ncomp = 4   ! a pretty cautious choice
	ELSE
	   ncomp = 1
	ENDIF
* ... if there are multiple transforms on the expression, each acts like
*     another component to be evaluated (at least I think so ...)
	ncomp = ncomp + n_dfrnt_xforms

* how much memory is there that is not already committed
	free_mem = max_mem_allowed
     .		- (permanent_mem + essential_mem + deferred_essential_mem)

* ... if the available memory is so small that even the result cannot fit, 
*     abort now where we can give the most informative error report
*     Note that IS_STRIP (and this routine) gets called up a second time
*     whenever an uncompressed axis has been chosen for splitting.  At that
*     time deferred_committed_mem will reflect the size of the result buffer
*     from the first pass.
	res_size = CGRID_UVAR_SIZE( res_cx )   ! size of result
	IF (free_mem .LE. res_size) GOTO 5100 

* how much memory to set aside as hedge against running out?
	IF (use_caution) THEN
	   set_aside_mem = (mode_arg(pmode_frugal,1)/100.) * free_mem
	   available_mem = free_mem - set_aside_mem
	ELSE
	   available_mem = free_mem
	ENDIF

* do we need to split up the calculation ?
	IF ( ((ncomp*src_size)+(nbuffers*res_size))
     .	 .LE. available_mem                           ) RETURN

****************************************************************
* ***** PHASE 2. Find the best single axis split    (skip 3 if yes)

* Init phase 2 variables
 100	nsplittable       = 0
	inadequate_split_length = .FALSE.

* Loop through the potential spliattable axes, giving preference to
* axes of slower access
	DO i = nferdims, 1, -1
	   IF ( dset.GT.pdset_irrelevant .AND. dset.LE.maxdsets ) THEN
              split_ax = ds_ax_speed(i, dset)
	   ELSE
	      split_ax = i
	   ENDIF

	   ax_splittable(split_ax) = .FALSE.  ! default until proven other

* Cannot split/gather on axes that are transformed with an algorithm
*       that depends upon looking at neighboring data points (well, not so
*       easily, anyway). smoothers, fillers, @LOC, etc.
	   ax_trans = cx_trans(split_ax,res_cx)
	   IF    (ax_trans .NE. trans_no_transform) THEN
	      IF (ax_trans .LE. trans_compress_code
     .       .OR. ax_trans .EQ. trans_locate        ) CYCLE
	   ENDIF

* Cannot split along a modulo regrid axis
	   IF ( ITSA_MODULO_REGRID(res_cx,split_ax) ) CYCLE

* Beware user-defined vars with limits specified in definition
	   IF ( itsa_uvar ) THEN
	      IF ( uvar_given(split_ax,var) .NE. uvlim_needed ) CYCLE
	   ENDIF

! 5/17 bug fix: was "axis" instead of "split_ax" here
	   IF (its_dsg) THEN
	      CALL POTENT_AX_LIMS(src_cx, split_ax, lo_ss, hi_ss)
	   ELSE
! 5/17 quite likely could get lims from src_cx, but if it aint broke ...
	      CALL POTENT_AX_LIMS(res_cx, split_ax, lo_ss, hi_ss)
	   ENDIF
	   split_ax_len = hi_ss - lo_ss + 1
	   IF (split_ax_len .EQ. 1) CYCLE

* The only compressed axes allowed for splitting are THE trans
* If there is more than one compressing transform (e.g. V[z=@AVE,T=@VAR])
* reordering them may give a wrong result due to non-linearity
* If res_cx includes other compressing transformations
* INTERP_STACK & IS_STRIP will return us here for those in turn
	   split_on_compressed_ax = ax_trans .GT. trans_compress_code
	   IF (split_on_compressed_ax
     .    .AND. .NOT.multiax_trans(split_ax)  )   CYCLE

* When there are multiple transformations in the expression and 
* multiple-axis transformations on the split axis, the gather 
* was returning only the last chunk of the split.

	   IF (split_on_compressed_ax .AND. n_dfrnt_xforms .GT. 1) CYCLE

* By how much does the source data exceed the available space?
* For a compressing calculation like @AVE
*     available_mem must hold   [nbuffers*res_size+src_size]  at one time
* If we split along the compressed axis there must be room for the result
* with its buffers and a chunk of source data reduced along the compressed axis
* ==> mem_used     = nbuffers*res_size + src_size/reduction_factor
* The maximum reduction factor is when we split into chunks of a single index
* ==> min_mem_used = nbuffers*res_size + src_size/split_ax_len
* If we split along an uncompressed axis, each chunk of the calculation
* acts like a smaller version of the full transformation, where the length
* of the split axis has been reduced
* ==> mem_used     = res_size + (nbuffers*res_size+src_size)/reduction_factor
* ==> min_mem_used = res_size + (nbuffers*res_size+src_size)/split_ax_len
	   rsrc_size = src_size    ! force calculations into floating point
	   IF (split_on_compressed_ax) THEN
	      min_mem_used = nbuffers*res_size
     .			   + (ncomp*rsrc_size)/split_ax_len
	   ELSEIF (its_dsg) THEN
	      min_mem_used = res_size
     .			   + (ncomp*rsrc_size)/split_ax_len
	   ELSE
! NOTE:  MODULO REGRIDDING has not ben thought through carefully
	      min_mem_used = res_size
     .			   + (nbuffers*res_size+ncomp*rsrc_size)/split_ax_len
	   ENDIF

* We found a splittable axis!
* If this axis is long enough to do the job, then use it
	   IF (available_mem .GE. min_mem_used ) GOTO 1000

* save a summary of this axis to use later if a multi-axis split is needed
	   nsplittable = nsplittable + 1
	   ax_splittable(split_ax) = .TRUE.
	   ax_compressed(split_ax) = split_on_compressed_ax
	   ax_len       (split_ax) = split_ax_len
	   ax_min_mem   (split_ax) = min_mem_used
	ENDDO

****************************************************************
* ***** PHASE 3. Find the best multiple axis split

* Done looping through the axes with no axis long enough to do the job, alone.
	inadequate_split_length = .TRUE.

* Now the hard case -- select the best two axes to split on:
* Splitting on multiple axes means that we will be returning here again
* When we split along the compressed axis the transform gets removed from
* the individual chunks, so further splitting of the chunk becomes impossible
* Therefore we must choose to split on an uncompressed axis first

* The optimal strategy is to choose the two slowest axes that together will
* provide the required requction factor.
* ax1 is the axis that will be chosen for splitting, so it must be uncompressed
	IF (nsplittable .GE. 2) THEN
	   DO i1 = nferdims, 1, -1
	      IF ( dset.GT.pdset_irrelevant .AND. dset.LE.maxdsets ) THEN
                 ax1 = ds_ax_speed(i1, dset)
	      ELSE
	         ax1 = i1
	      ENDIF

	      IF (.NOT.ax_splittable(ax1)) CYCLE
	      IF (ax_compressed(ax1)) CYCLE

	      DO i2 = nferdims, 1, -1
	         IF ( dset.GT.pdset_irrelevant .AND. dset.LE.maxdsets ) THEN
                    ax2 = ds_ax_speed(i2, dset)
	         ELSE
	            ax2 = i2
	         ENDIF

	         IF (ax2 .EQ. ax1) CYCLE
	         IF (.NOT.ax_splittable(ax2)) CYCLE

* ... can these two axes together provide the needed reduction in size?
*     the minimum memory needed for the 2-level nested gather is
*     ax_min_mem(ax1) + ax_min_mem(ax2)/ax_len(ax1)
	         IF (available_mem .GE. ncomp*
     .		     (ax_min_mem(ax1) + ax_min_mem(ax2)/ax_len(ax1))  ) THEN
* ... we found a 2-level split/gather that will work
	            split_ax = ax1
	            split_on_compressed_ax = .FALSE.
	            GOTO 1000
	         ENDIF
	      ENDDO
	   ENDDO
	ENDIF

* We did not find two axes that can fit this transformation into avail_mem
* Try again with less caution, if wiggle room still exists
	IF (use_caution) GOTO 2000

* Boy, we're in a tough spot. 
* No answer was found that was guaranteed to fit into available_mem
* Instead choose a strategy that represents a reasonable college try
* ... Whatever axis we choose, at this point we are guaranteed that the frag
*     size will be 1 (else we wouldn;t be at this point)
*     so there is no reason to adjust available_mem back to its cautious level
* Choose an uncompressed axis if one is available, because it can be
* part of a 2-axis split.  Choose the longest one.  It reduces the most.
	no_axis_found = .TRUE.
	DO i = 1, nferdims
	   IF (.NOT.ax_splittable(i)) CYCLE
	   IF (ax_compressed(i)) CYCLE
* ... found an uncompressed axis
	   IF (no_axis_found) THEN
	      split_ax = i
	      no_axis_found = .FALSE.
	   ELSEIF (ax_len(i) .GT. ax_len(split_ax)) THEN
	      split_ax = i
	   ENDIF
	ENDDO

* If no uncompressed axis is available, then choose the longest compressed axis
	IF (no_axis_found) THEN
	   DO i = 1, nferdims
	      IF (.NOT.ax_splittable(i)) CYCLE
* ... found an uncompressed axis
	      IF (no_axis_found) THEN
	         split_ax = i
	         no_axis_found = .FALSE.
	      ELSEIF (ax_len(i) .GT. ax_len(split_ax)) THEN
	         split_ax = i
	      ENDIF
	   ENDDO
	ENDIF

* skunked?
	IF (no_axis_found) RETURN ! mod regrid w no other axes available?

	split_on_compressed_ax = ax_compressed(split_ax)
!	GOTO 1000

****************************************************************
* ***** PHASE 4. Setup the split/gather

* OK - lets divvie this baby up into chunks!
* save the default context so it can be re-used at each chunk
* a compressed axis is represented as lo/hi_ss=-999, which is not what we want
* so if splitting on a compressed axis, store src_cx
 1000	axis = split_ax
	IF (split_on_compressed_ax .OR. its_dsg) THEN
* ... when splitting on a compressed axis the stack activity remains the 
*     compressing transform.  It is IS_TRANS, instead of IS_GATHER, that
*     will control the fetching of fragment after fragment
	   CALL CREATE_NEW_CONTEXT( src_cx, big_cx, status )	! pushes stack
	   is_phase( lo_isp ) = isphase_initialize_transform
	ELSE
	   aclass = isact_class_gather
	   is_act(lo_isp) = isact_gather
	   CALL CREATE_NEW_CONTEXT( res_cx, big_cx, status )	! pushes stack
	ENDIF
	IF ( status .NE. ferr_ok ) GOTO 5000
	is_big_cx( lo_isp ) = big_cx
* cancel result pointer to indicate no result yet
	is_mr( lo_isp ) = unspecified_int4
* determine the number of subscripts that can be processed in each chunk
	IF (its_dsg) THEN
	   CALL POTENT_AX_LIMS(src_cx, axis, lo_ss, hi_ss)
	ELSE
! 5/17 quite likely could get lims from src_cx, but if it aint broke ...
	   CALL POTENT_AX_LIMS(res_cx, axis, lo_ss, hi_ss)
	ENDIF
	split_ax_len = hi_ss - lo_ss + 1
	IF (split_on_compressed_ax) THEN
	   reduction_factor_needed = ncomp*rsrc_size
     .				   / (available_mem-nbuffers*res_size)
	ELSE
	   reduction_factor_needed = (nbuffers*res_size + ncomp*rsrc_size)
     .				   / (available_mem-res_size)
	ENDIF
	frag = split_ax_len / reduction_factor_needed
	IF (frag .LT. 1) frag = 1
	is_sum( lo_isp ) = frag
* set the subscript range for the first gather fragment
	CALL TRANSFER_CONTEXT( big_cx, src_cx )
	cx_hi_ss(src_cx,axis) = lo_ss + frag - 1
	cx_by_ss(axis,src_cx) = .TRUE.
	CALL FLESH_OUT_AXIS( axis, src_cx, status )
	IF ( status .NE. ferr_ok ) GOTO 5000

* diagnostic output about split on compressed axis
*    (this is a "dummy" interp stack activity in the sense that is_act is
*     never actually given this activity.  It exists only to generate an
*     informative diagnostic report)
	IF ( mode_diagnostic
     .	.AND. (split_on_compressed_ax.OR.its_dsg) ) THEN
	   CALL DIAG_OP('strip', isact_class_cmp_gather_dummy, big_cx, axis)
	   CALL DIAG_OP('doing', isact_class_cmp_gather_dummy, src_cx, axis)
	ENDIF

* at this point we have obligated the use of memory for the result,
* but we have not yet actually created the memory objects.
* That will occur in IS_TRANS during its initiailzation phase.
* So here we plant a record of what we have obligated into the stack, so that
* when multi-axis strip-gather operations are needed, the next (nested) call
* to this routine will know of the deferred memory allocations
* We only alert to deferred memory usage if the frag we have selected is
* inadequate to fillfill the memory reduction with a single axis split
	IF (inadequate_split_length) THEN
	   IF (split_on_compressed_ax) THEN
	      is_deferred_mem(lo_isp) = nbuffers*res_size
	   ELSE
	      is_deferred_mem(lo_isp) = res_size
	   ENDIF
	ELSE
	      is_deferred_mem(lo_isp) = 0
	ENDIF

* store diagnostic information about the splitting (see SHOW MEM split)
	IF (nsplits .LE. max_splits) THEN
	   IF (nsplits .EQ. 0) THEN
	      dup_split = .FALSE.
	   ELSE
	      dup_split = var     .EQ. split_var  (nsplits)
     .             .AND. cat      .EQ. split_cat  (nsplits)
     .             .AND. trans    .EQ. split_trans(nsplits)
     .             .AND. split_ax .EQ. split_axis (nsplits)
	   ENDIF
	   IF (dup_split) THEN
	      split_repeats(nsplits) = split_repeats(nsplits) + 1
	   ELSE
	      nsplits = nsplits + 1
	      split_var  (nsplits) = var
	      split_cat  (nsplits) = cat
	      split_axis (nsplits) = split_ax
	      split_trans(nsplits) = trans
	      split_frag(nsplits)  = frag
	      split_repeats(nsplits) = 1
	   ENDIF
	ENDIF

* successful completion
	RETURN

* prepare to go back for another pass
* ... pass 1 tries to succeed with mode_frugal percent held in reserve
* ... pass 2 tries again half that much in reserve
* ... pass 3 tries again using all available memory
 2000	available_mem = available_mem + 0.500001*set_aside_mem
	IF (available_mem .GE. free_mem) THEN
	   available_mem = free_mem
	   use_caution = .FALSE.
	ENDIF
	GOTO 100

* error exits
 5000	RETURN
 5100	CALL ERRMSG( ferr_insuff_memory,status,
     .	    'Requested result is larger than the available memory', *5110)
 5110	CALL SHOW_ABORTED_MEM_STATE(res_size)
	GOTO 5000

	END
