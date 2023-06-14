	SUBROUTINE IS_AUXVAR_REGRID(  *, *, status )

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
* this routine controls the fetching of auxiliary variables if more than
* one; the determination of the curvilinier index region and fetching of
* the target; and (optionally) performing the regridding.  The regridding
* step is skipped when this routine is called under isact_aux_plot_vars
*
* regridding:  regrid the data from the source grid to the destination grid
* using auxiliary variables to guide the numerics of the regrid
* architype is a conversion from density to depth "TEMP[gz(depth)=zax]"
* where "depth" is a field of depths defined on the same density axis
* as the variable TEMP

* programmer - steve hankin
* NOAA/PMEL, Seattle, WA - Tropical Modeling and Analysis Program
* written for VAX computer under VMS operating system
*
* revision 0.0 - 9/2013 based upon IS_REGRID
* for Ferret v685+
* v695+ *sh* 3/15 - support for TF (FMRC) regridding added
* V698 *sh* 1/16 - bug fix: cx_regrid_trans were getting clobbered in dst_cx
* 11/2022 *acm* Code cleanup: Modernize DO-loop syntax. 

	IMPLICIT NONE

	include 'tmap_dims.parm'
	include	'ferret.parm'
	include	'errmsg.parm'
	include	'interp_stack.parm'
	include	'xtext_info.cmn'
	include	'xprog_state.cmn'
	include	'xcontext.cmn'
	include	'xvariables.cmn'
	include	'xalgebra.cmn'   ! with alg_regrid_code

* calling argument declarations:
	INTEGER	status
* normal exit is for error conditions
* exit #1 is to signal completion of the activity at this stack level

* internal variable declarations:
	LOGICAL	ITS_UN_CACHED, not_reusable, first, aux_has_cell_pts
	INTEGER TM_LENSTR1, CAXIS_LEN, AUX_AXIS_FROM_ARG,
     .		i, j, icom, idim, mr, lo_cx, hi_cx, lo_isp, hi_isp, naxis,
     .		aux_mr_list(nferdims), mr1, mr2, mr3,
     .		aux_cx_list(nferdims), cx1, cx2, cx3,
     .		isp_ptr, axis_list(nferdims),
     .		dst_cx, src_cx, com_cx, dst_mr, src_mr, frame_bottom, 
     .		impl_grid, slen, s1, sn, naux, ptr
	CHARACTER*128 VAR_CODE, src_name, com_name

!* temporary cuz totalview doesn't see equiv vars
!	INTEGER im1, im2, im3, ic1, ic2, ic3 

        EQUIVALENCE
     .          ( aux_mr_list(1), mr1 ),
     .          ( aux_mr_list(2), mr2 ),
     .          ( aux_cx_list(1), cx1 ),
     .          ( aux_mr_list(2), mr2 ),
     .          ( aux_cx_list(2), cx2 ),
     .          ( aux_mr_list(3), mr3 ),
     .          ( aux_cx_list(3), cx3 )


* this routine guides the regridding process through three phases:
*  1) isphase_get_aux_var  - gets the needed auxiliary vars onto the cx stack
*  2) isphase_get_source_var - gets the pre-regridded data field
*  3) isphase_do_aux_regrid  - performs the regridding

* RETURN 1 is to get another component
* RETURN 2 is when the regridding is completed
* normal RETURN is for errors

* auxiliary regridding always requires at least 2 component variables:
*	1 or more auxiliary variables  (e.g. xcoords, and ycoords for curvilinear regrid)
* and   the source data to be regridded

* for each components obtained, is_obj(isp) will retain a pointer to the
* parent level -- the level at which regridding was requested (a.k.a. "lo_isp")

* in all 3 phases the interp stack setup will look like this
*	stack level		stack contents
*       -----------		--------------
*	isp+1	<--	level  of component just obtained
*	...     <--     levels of components previously obtained
*	lo_isp	<--	level at which regrid was requested
* --- end of introductory code ---

* initialize
	IF ( is_phase(isp).EQ. isphase_get_aux_var
     . .AND. is_sum(isp) .EQ. 1                     )  THEN
* ... first time called.  Set up is_obj pointer
	   lo_isp = isp
	   is_obj(isp+1) = lo_isp ! so auxvar just obtained points to regrid op
	ELSE
	   lo_isp   = is_obj(isp)    ! the isp level of the regrid operation
	ENDIF
	lo_cx    = is_cx(lo_isp)
	naux    = cx_naux( lo_cx )
	IF (cx_type(lo_cx) .EQ. ptype_string) GOTO 5010

* set pointers to the auxiliary variables
* (if we are have more aux variables to get some of these pointers will be wrong)
* ... first pre-fill with known values
	DO i = 1, 3
	   aux_mr_list(i) = dummy_mr
	   aux_cx_list(i) = unspecified_int4 
 	ENDDO

        i = 0
        DO idim = 1, nferdims
           IF (cx_aux_var(idim,lo_cx) .NE. unspecified_int4) THEN
              i = i + 1
              isp_ptr = lo_isp + i 
              axis_list(i) = idim
	      IF (idim .EQ. f_dim
     .	    .AND. cx_aux_var(t_dim,lo_cx) .NE. unspecified_int4) THEN
* ... note that with TF regridding there are 2 dimensions with cx_aux_var but
*     only one auxiliary variable, TIME2D, so both axes point to same aux var
                 aux_mr_list(i) = aux_mr_list(1)
                 aux_cx_list(i) = aux_cx_list(1)	         
	      ELSE
                 aux_mr_list(i) = is_mr(isp_ptr)  ! see equivs
                 aux_cx_list(i) = is_cx(isp_ptr)  ! see equivs
	      ENDIF
          ENDIF
	ENDDO

* ... TF-regridding (FMRC) introduces the idea that the number of axes being
*     regridded (T&F) may differ from the number of aux variables (time2d)
	naxis = i

* sanity checks
!        IF ( naux.GT.1 )
!     .		CALL ERRMSG(ferr_prog_limit, status,
!     .		'for now TF is the only 2 aux var regrid supported', *5000)

	IF ( is_phase(lo_isp) .EQ. isphase_get_aux_var ) THEN 

* we're in the initial phase getting the components needed
	   CALL CREATE_NEW_CONTEXT( lo_cx, hi_cx, status )	! pushes stack

* put isp up one so that the component just received stays on the stack
	   isp = isp + 1

	   IF (is_sum(lo_isp) .LT. naux ) THEN
*  *** SET UP TO GET THE NEXT AUXILIARY VARIABLE ***
	      is_sum(lo_isp) = is_sum(lo_isp) + 1  ! the next auxvar
	      CALL GET_AUX_VAR_CONTEXT( lo_cx, hi_cx, is_sum(lo_isp),
     .					status )
	      IF ( status .NE. ferr_ok ) GOTO 5000
	      is_axis(lo_isp) = AUX_AXIS_FROM_ARG(lo_cx, is_sum(lo_isp) )

	   ELSE
* *** SET UP TO GET THE SOURCE DATA ***
	      is_phase(lo_isp) = isphase_get_source_var

* set up 4D array limits as unsubscripted variables in COMMON
              CALL NON_ARRAY_SUBSC( aux_mr_list, naux )

* create the context for the source data needed to perform the regrid
              CALL AUX_REGRID_LIMS( lo_cx, hi_cx, axis_list, naxis,
     .                memry(mr1)%ptr, mr1, cx1,
     .                memry(mr2)%ptr, mr2, cx2,
     .                memry(mr3)%ptr, mr3, cx3,
     .                status )
              IF (status .NE. ferr_ok) GOTO 5000
	   ENDIF

* go get the next component (the one just set up) via RETURN 1
   	   CALL PUSH_INTERP_STACK( isp, status )
	   IF ( status .NE. ferr_ok ) RETURN
	   is_cx ( isp ) = hi_cx
	   is_obj( isp ) = lo_isp  ! points back to the regridded variable
	   IF ( cx_category(hi_cx) .EQ. cat_user_var ) THEN
* ... setup to evaluate auxiliary var that is user-defined (uvar)
	      is_uvar( isp ) = cx_variable( hi_cx )
	   ENDIF
	   RETURN 1

	ELSEIF (  is_phase(lo_isp) .EQ. isphase_get_source_var ) THEN 
 
* ***  TIME TO REGRID OR RETURN "3 ARGUMENTS" (2 for Z)  FOR PLOTTING****
* ... we have all of the needed components.
	   is_phase(lo_isp) = isphase_do_aux_regrid  ! (never used)

* if the scenario is
*       yes? LET TEMPZ = TEMP[gz(DEPTH)=zax]
*       yes? LOAD TEMPZ
*    then the interp stack setup at the time this is called is
*
*   stack level            stack contents
*   ----- -----      -----------------------------------
* hi_isp:isp+naux   variable on native grid (TEMP)
*        isp        auxiliary variable (DEPTH)
* lo_isp:isp-1      requested regrid result (TEMP[dz(DEPTH)=zax])
*        isp-2      evaluating TEMPZ
*        isp-3      evaluating EXP#


  	   hi_isp = isp + cx_naux(lo_cx)   ! the source data (just obtained)
	   src_cx = is_cx(hi_isp)
	   dst_cx = lo_cx
	   src_mr = is_mr(hi_isp)

*******************************
* RETURN AUX VARS FOR "3 ARGUMENT" PLOTS
* No regridding to do.
* Transfer the isp stack contents of the component variable (hi_isp)
* to the destination level (lo_isp) and reset the stack pointer
* Park the auxiliary variables in the special parking area at the stack top
* XXX TEMPORARY: tbd - also park a copy of their cx at the top of the cx stack?
	   IF (is_act(lo_isp) .EQ. isact_aux_plot_vars) THEN
* park the auxiliary variable(s) at the top of the isp stack
	      i = 0
	      DO idim = 1, nferdims
	         IF (cx_aux_var(idim,lo_cx) .NE. unspecified_int4) THEN
* XXX TEMPORARY - warn of same auxvar parked twice.  Better would be to ERROR
*	when saved context regions and mr don't match btwn now and earlier
	            ptr = is_aux_ptr(lo_isp) + idim
	            IF (is_mr(ptr) .NE. unspecified_int4)
     .			CALL WARN("is_auxvar_regrid dup auxvar parked")
	            CALL TRANSFER_ISP(isp+i, ptr)		! save it
	            is_axis(ptr) = idim			        ! bookkeeping
* ... also park a copy of the aux var context at the top of the cx stack
	            is_cx(ptr) = max_context+idim
	            CALL TRANSFER_CONTEXT(is_cx(isp+i),is_cx(ptr))
* ... set the aux axis limits to the limits required getting the target var
	            CALL TRANSFER_AXIS(idim, src_cx, is_cx(ptr))
	            i = i + 1
	         ENDIF
	      ENDDO
	      IF (i .NE. cx_naux(lo_cx)) CALL WARN("is_auxvar_regrid crptn")
* the component just obtained is *the* result (no regridding to do)
	      CALL TRANSFER_CONTEXT( src_cx, dst_cx )
	      is_mr(lo_isp) = src_mr
	      CALL MR_IN_USE( src_mr )  ! hack cuz we will "un-use" it below
	      GOTO 400     ! clean up the stack
	   ENDIF
*******************************

* Make the destination context like the source data context on all except
* the regrid axis.  That one keeps the user-requested size.
* (other axes may have been modified by information embedded within a
* user-variable definition.) 

* (*sh* untested, simpler(?) alternative is to create the mr from dst_cx just
* as it is, pre-fill with missing, and create cx_valid based on available
* limits of the source data and the auxiliary variable)

* insist that on the axes being regridded the size of the auxiliary fields is
* either the same as the source data or +1 larger.  +1 indicates cell boundary
* coordinates in the auxiliary variable(s).  Determine whether point locations
* or bounds were provided.
	   first = .TRUE.
	   DO i = 1, naxis
	      idim = axis_list(i)
* ... check all relevant aux vars
	      DO  j = 1, naxis 
	         com_cx = aux_cx_list(j)
	         IF (CAXIS_LEN(idim,com_cx) .NE. CAXIS_LEN(idim,src_cx)+1
     .	       .AND. CAXIS_LEN(idim,com_cx) .NE. CAXIS_LEN(idim,src_cx) )
     .									GOTO 5500
	         IF ( first ) THEN
	            aux_has_cell_pts = CAXIS_LEN(idim,com_cx)
     .			          .EQ. CAXIS_LEN(idim,src_cx)
	            first = .FALSE.
	         ELSE
	            IF (aux_has_cell_pts .NEQV. 
     .			CAXIS_LEN(idim,com_cx).EQ.CAXIS_LEN(idim,src_cx) )
     .									GOTO 5500
	         ENDIF
	      ENDDO
	   ENDDO

	   CALL TRANSFER_CONTEXT( src_cx, cx_buff )
	   cx_unstand_grid( cx_buff ) = cx_unstand_grid( dst_cx )
	   cx_grid        ( cx_buff ) = cx_grid        ( dst_cx )
	   cx_impl_grid   ( cx_buff ) = cx_impl_grid   ( dst_cx )
	   cx_naux        ( cx_buff ) = cx_naux        ( dst_cx )
           DO idim = 1, nferdims
              cx_aux_var (idim, cx_buff) = cx_aux_var (idim, dst_cx)
              cx_aux_cat (idim, cx_buff) = cx_aux_cat (idim, dst_cx)
              cx_aux_stat(idim, cx_buff) = cx_aux_stat(idim, dst_cx)
	      cx_regrid_trans(idim,cx_buff ) = cx_regrid_trans(idim, dst_cx)
	      cx_delta(idim,cx_buff) = unspecified_val8
	   ENDDO
	   DO i = 1, naxis 
	      idim = axis_list(i)
	      CALL TRANSFER_AXIS( idim, dst_cx, cx_buff )
	      cx_regrid_trans(idim, cx_buff) = cx_regrid_trans(idim, dst_cx)
	      cx_delta(idim,cx_buff) = unspecified_val8
	   ENDDO
	   CALL TRANSFER_CONTEXT( cx_buff, dst_cx )  ! (nervous. is it ok to clobber this?)

* create a space for the new result possibly with the edges flagged as bad
	   CALL CREATE_MEM_VAR( dst_cx, dst_mr, status )
	   IF ( status .NE. ferr_ok ) GOTO 5000
	   is_mr( lo_isp ) = dst_mr

* auxiliary regridding often creates ragged edges, so pre-fill with missing values
           CALL FILL_MEMORY ( dst_mr, mr_bad_data(dst_mr) )

* diagnostic message ?
	   IF (mode_diagnostic) THEN
	      CALL DIAGNOSTIC_OUT('auxgrd ',dst_mr,  point_to_mr)
	   ENDIF 

* ... set up 4D array limits as unsubscripted variables in COMMON
	   aux_mr_list(4) = src_mr
	   aux_mr_list(5) = dst_mr
           CALL NON_ARRAY_SUBSC( aux_mr_list, 5 )    ! 5 arguments to pass

	   CALL DO_AUX_VAR_REGRID(  axis_list, naxis, aux_has_cell_pts,
     .	   		memry(src_mr )%ptr, src_mr,  src_cx,
     .			memry(dst_mr )%ptr, dst_mr,  dst_cx,
     .			memry(mr1)%ptr, mr1, cx1,
     .			memry(mr2)%ptr, mr2, cx2,
     .			memry(mr3)%ptr, mr3, cx3,
     .			status )

	   IF (status .NE. ferr_ok) GOTO 5000

* clean up 
 400	   CONTINUE
* (the following code was adapted from IS_DO_OP)

* the "frame_bottom" is the isp level that will be left when the components of
* the current calculation (1 to 3) are reduced to a single value.
* on exit isp will point to the frame bottom and the result will be left there
	   frame_bottom = lo_isp

* ... if any components were un_cached we must uncache the result
	   not_reusable = .FALSE.
	   DO icom = isp, frame_bottom, -1
	      IF ( ITS_UN_CACHED(is_mr(icom)) ) not_reusable = .TRUE.
	   ENDDO
	   IF ( not_reusable ) CALL UN_CACHE( dst_mr )

* ...release component cx stack space
	   DO isp = hi_isp, frame_bottom, -1
	      IF ( is_cx(isp) .NE. cx_stack_ptr ) WRITE (6,*) 'OP_STK??'	! temp
	      cx_stack_ptr = cx_stack_ptr - 1
	      mr  = is_mr( isp )
	      IF ( mr_protected(mr) .EQ. mr_temporary
     .	      .OR. mr_category (mr) .EQ. cat_temp_var ) THEN
	         CALL DELETE_VARIABLE( mr )
	      ELSE
	         CALL MR_NOT_IN_USE( mr )
	      ENDIF

* ...deallocate dynamic grids from components
	      com_cx = is_cx(isp)
	      IF ( cx_has_impl_grid(com_cx) ) THEN
	         impl_grid = cx_grid(com_cx)
	         CALL DEALLO_DYN_GRID(impl_grid)
	         cx_has_impl_grid(com_cx) = .FALSE.
	      ENDIF
	   ENDDO

* collapse the interp stack frame
	   isp = frame_bottom

* save the result context at the cx level previously used by the 1st component (why?)
	   cx_stack_ptr = cx_stack_ptr + 1
	   CALL TRANSFER_CONTEXT( dst_cx, cx_stack_ptr )
	   is_cx( isp ) = cx_stack_ptr

	   RETURN 2
	ENDIF

* error exits
 5000	RETURN
 5010	CALL ERRMSG( ferr_invalid_command, status,
     .		'cant do auxiliary variable regridding on string variables',
     .		 *5000)
 5500	com_name = VAR_CODE(cx_category(com_cx),
     .                      cx_variable(com_cx) )
        src_name = VAR_CODE(cx_category(src_cx),
     .                      cx_variable(src_cx) )
        i = TM_LENSTR1(com_name)
        j = TM_LENSTR1(src_name)
        CALL ERRMSG( ferr_invalid_command, status,
     .          'auxiliary variable '//com_name(:i)//
     .          ' has non-conformable length on '//ww_dim_name(idim)//
     .          ' axis for regridding '//src_name(:j), *5000 )
	END	





