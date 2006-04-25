	SUBROUTINE INTERP_STACK ( memory, isp_base, dflt_cx,
     .				  name_code, status )

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
* Return a grid of data and a buffer of context info about the grid given 
* a default context for evaluation and the text of an expression.
* Transformations, regrids, rotations, etc. are all controlled
* by this routine.  Nesting of these operations is possible because all
* operations are carried out on a stack - the "interpretation stack".
* The results are left on the interpretation stack at level isp_base
* The result computed, pointed to by is_mr(isp_base), is left protected

* programmer - steve hankin
* NOAA/PMEL, Seattle, WA - Tropical Modeling and Analysis Program
* written for VAX computer under VMS operating system
*
* revision 0.0 - 2/10/87
* .
* .
* .
* V200:  5/18/89 - 4D-symmetrical
*		 - implicit regridding
*		 - transformations not permitted on command line
*		 - eliminated get_ctxt_only from call
*		 - "cvar" evaluation is done on the IS stack
*		 - uvar grids determined in IGRID
*	 1/11/90 - added @LOC
*	 2/28/90 - added kludge code to double check memory before IS_READ
* for Unix/RISC port 9/91 - added Hanning,Parzen and Welch filters (windows)
*                         - added CLEAN_STACK for better error handling
* V230: 12/26/91 - added @SUM, @WEQ, @RSUM
*        1/22/92 - added @WEQ
*         2/3/92 - added @VAR, @NGD, and @NBD
*        3/23/92 - added @FLN (fill with linear interpolation)
* V312: 5/94  - array "memory" as a calling argument
* V320: 12/94 - added logic to trap modulo reads on file variables
*	 2/95 - added logic to trap reverse-direction axes and reverse them
* V420: 11/95 - moved modulo logic from file vars only to all variables
*		(12/95) screen out uvlim_*_xact from modulo
*	      - added isact_class_reconcile so cmnd qualifiers are
*		better processed
*	      - support for non-reusable results: if component is non-reusable
*		so is result (using ITS_UN_CACHED and UN_CACHE)
* V430: 7/96  - incorporate 4D transformations
* V450: 2/97  - Allow application of modulo to GC vars (but not other uvars)
* V455    9/97 - added transforms @CDA,CDB,CIA,CIB
* V500: 10/98 *jc* - added lines to ignore hidden @RDC
*	12/98 *sh* added call to IS_CONST_VAR to support {1,3,5} notation
*	 3/99 *sh* added call to COMPLETE_MISSING_LIMITS to improve cache hits
*		and corresponding change in file reading to restrict limits
*	 5/99 *sh* remove delta values error trap
*		(cx_delta is now used by netCDF vars to signal that the
*		read_tm should use strides)
* V510: 8/99 *sh* handle isact_sample
* V530: 11/00 *sh* - perform RESTRICT_FILE_LIMITS before recheck FIND_MEM_VAR
*		for improved cache hits (case; ASCII file reads where EZ axis
*		exceeds actual file size misses cache because
*		COMPLETE_MISSING_LIMITS inserts full EZ ax limits)
* V533: 6/01 *sh* - added @EVNT
* V541: 2/02 *sh* - added support for modulo lengths (subspan modulo)
* V554 1/04 *acm* - additions for /RANGE repeat loop control

	include	'ferret.parm'
	include	'errmsg.parm'
	include	'interp_stack.parm'
	include	'xcontext.cmn'
	include	'xvariables.cmn'
	include	'xinterrupt.cmn'
	include	'xfr_grid.cmn'

* calling argument declarations:
	INTEGER		isp_base, dflt_cx, status
	REAL		memory(*)
	CHARACTER*(*)	name_code

* internal variable declarations:
	LOGICAL		ITS_UN_CACHED, ITSA_MODULO_REGRID, ITSA_GCVAR,
     .			ITSA_MODULO_GCF, TM_ITS_SUBSPAN_MODULO,
     .			look_again, modulo_axes(4), explicit_qualifiers
	INTEGER		UVAR_NUMBER, KNOWN_GRID, CGRID_AXIS, 
     .			CAXIS_MODLEN, TM_AXES_REVERSED, rev_axes(4),
     .			cx, mr, idim, cat, act, uvar_pointer,
     .			com_mr, var

* external routines passed
	EXTERNAL	DO_SHIFT, DO_SMTH_BOX, DO_FILL_AVE, DO_AVERAGE,
     .			DO_INTEG_DEF, DO_4D_AVE, DO_4D_INT_DEF,
     .			DO_DERIV_FRWD, DO_DERIV_CNTR,DO_DERIV_BKWD,
     .			DO_SMTH_BINML, DO_MIN, DO_MAX, DO_INT_INDEF,
     .			DO_INTERPOLATE, DO_LOCATE, DO_SMTH_HANNG,
     .                  DO_SMTH_PARZN, DO_SMTH_WELCH, DO_SUM,
     .                  DO_DIST_LOC, DO_RUN_SUM, DO_NGOOD, DO_NBAD,
     .                  DO_VARIANCE, DO_FILL_INTERP, DO_FILL_NEAR,
     .			DO_4D_VAR,DO_4D_NGD,DO_4D_NBD,
     .			DO_NEAREST_DIST_ABOVE, DO_NEAREST_DIST_BELOW,
     .			DO_NEAREST_INDEX_ABOVE,DO_NEAREST_INDEX_BELOW,
     .			DO_EVENT

* calling arguments:
* in	isp_base	- stack level at which to begin calculations
*			  ( all levels below remain untouched )
* in	dflt_cx		- "default context" for evaluation of expression
* in	name_code	- name of uvar to be evaluated
* out	status		- completion code

* example of stack operations:
*	let DTDX = ( temp[i=+1d] - temp[i=-1d] ) / 2*dx
* ... and we wish to evaluate
*	LIST/X=170E:160W:10 U[G=TEMP,Z=@AVE] * DTDX

* outside of this routine the default context (from SET REGION,DATA,etc)
* has been modified by "/X=170E:160W:10" to produce dflt_cx
* Before entering this routine the text of the expression has been moved to 
* a numbered buffer in uvar_text and broken down into reverse Polish 
* ordered operations within other arrays in the uvar structure.

* When the stack is entered ( in isact_request mode ) it will detect the delta
* 10 degrees specified on the X axis and interpret this as an implicit request
* to regrid the data.  IS_STRIP will be called to strip this off by claiming
* a level on the IS stack for the regridding operation, incrementing the IS
* stack pointer, isp, and passing on the context (pointed to by is_cx)
* modified by the removal of ":10".

* Having no transformations or regrids in its current top level context 
* INTERP_STACK will then treat the context as an algebraic expression and
* IS_ALGEBRA will be given control.  IS_ALGEBRA will claim an IS stack level
* for its operations.  IS_ALGEBRA will parse the string "U[G=TEMP]",
* identifying data set, variable, grid, transformations, etc. and setting
* up a context that describes it.  The resulting request will then be passed
* back to INTERP_STACK at label 10 in IS_REQUEST mode.

* INTEEP_STACK will then detect the request for Z averaging.  Again,
* IS_STRIP will be called to allocate a stack level for the operation and
* strip the averaging request off in the context for the next stack level.

* INTEEP_STACK will then detect the explicit request for regridding and call
* IS_STRIP again to allocate a stack level for the operation and strip the
* regridding request off in the context for the next stack level.

* INTERP_STACK, again in IS_REQUEST mode, will then see that the context at the
* top of the stack now contains a request for a variable ("U") to be read and
* it will call IS_READ.  The variable will be read and the IS stack pointer
* will be decremented.

* INTERP_STACK now finds itself in regridding mode (isact_regrid) and calls
* IS_REGRID to perform the operation on the results just averaged.  When
* the regridding operation is completed isp is decremented.

* INTERP_STACK now finds itself in averaging mode (isact_average) and calls
* IS_AVERAGE to perform the operation on the results just read.  Again, when
* the averaging operation is completed the IS stack pointer,isp, is decremented.

* INTERP_STACK now finds itself back in algebra mode (isact_algebra) and calls
* IS_ALGEBRA giving it the requested component pointed to by is_mr on the IS
* stack.  IS_ALGEBRA then moves on to the next component, DTDX, parses the
* expression and passes the request back to INTERP_STACK. ... etc.
* When IS_ALGEBRA regains control again it will have the 2 needed operands
* on the stack and it will perform the requested multiplication operation.
* It will then detect that it has reached the end of the expression and
* decrement isp.

* INTERP_STACK now finds itself back in implicit regrid mode.  It calls
* IS_AUTOGRID to perform the required action.  IS_AUTOGRID decrements isp
* when it is finished.

* INTERP_STACK then finds itself back at the base IS stack level signaling 
* that the expression has been evaluated successfully.  It exits signalling
* successful completion.

* --- end of introductory code ---

* locate the requested variable in the uvar variables
	uvar_pointer = UVAR_NUMBER( name_code )
	IF ( uvar_pointer .EQ. unspecified_int4 ) CALL ERRMSG
     .		( ferr_internal, status, 'intp_stk', *5000 )
	
* initialize the default context for the quantity to evaluate
	cx_variable( dflt_cx ) = uvar_pointer
	cx_category( dflt_cx ) = cat_user_var
	cx_grid    ( dflt_cx ) =
     .		KNOWN_GRID(cx_data_set(dflt_cx),cat_user_var,uvar_pointer)

* initialize base level of the interpretation stack
* ... ( see PUSH_INTERP_STACK for isp>isp_base )
 6	isp = isp_base			! stack pointer
	is_cx  ( isp ) = dflt_cx	! context for stack activities
	is_uvar( isp ) = uvar_pointer	! pointer to text buffer

* when called the request is ALWAYS for an untransformed implicitly defined
* user variable (e.g. "LIST U" or "LIST SIN(X+Y)"
* determine its grid, if necessary - then flesh out the requested dimensions
	IF ( cx_grid(dflt_cx) .EQ. unspecified_int4 ) THEN
	   CALL GET_UVAR_GRID( uvar_pointer,cx_data_set(dflt_cx),status )
	   IF ( status .NE. ferr_ok ) GOTO 5000
	   cx_grid(dflt_cx) =
     .		KNOWN_GRID(cx_data_set(dflt_cx),cat_user_var,uvar_pointer)
	ENDIF

* flesh out the requested dimensions (noting if there are any)
	explicit_qualifiers = .FALSE.
	DO 5 idim = 1, 4
	   IF ( uvar_given( idim, uvar_pointer ) .GE. uvlim_needed
     .	   .OR. cx_given( idim,dflt_cx ) ) THEN
	      explicit_qualifiers = explicit_qualifiers
     .			       .OR. cx_given(idim,dflt_cx )
	      CALL CONFINE_AXIS( idim, dflt_cx, status )
	      IF ( status .NE. ferr_ok ) GOTO 5000
	      CALL FLESH_OUT_AXIS( idim, dflt_cx, status )
	      IF ( status .NE. ferr_ok ) GOTO 5000
	   ENDIF
 5	CONTINUE
	
* was a regridding operation implied through the command qualifier cx_delta?
* (5/99) test moved from transformation stripping section below
	DO 8 idim = 1, 4
	   IF ( cx_delta(idim, dflt_cx) .NE. unspecified_val8 ) THEN
	      CALL ERRMSG( ferr_invalid_command, status,
     .			'delta values may only modify variable names', *5000 )
	   ENDIF
 8	CONTINUE



* special pre-initialization of stack to reconcile differences between command
* qualifiers and expression -- e.g. LIST/I=1:5 a[i=3:4]
	IF ( explicit_qualifiers ) THEN
	   CALL IS_STRIP( isact_class_reconcile, 0, *10, status )
	   GOTO 5000
	ENDIF

******************************************************************************
* * * * * * TOP OF INTERPRETATION STACK - INITIALIZE STACK LEVEL * * * * * *
******************************************************************************
 10	is_act(isp)   = isact_request
	cx = is_cx( isp )
	cat = cx_category( cx )
	var = cx_variable( cx )

* constants and pseudo-variables and counter variables may be handled immediately
* ... get user variable grids if unknown
	IF     ( cat .EQ. cat_pseudo_var ) THEN
	   CALL IS_PSEUDO ( memory, *2000, status )
	   GOTO 5000
	ELSEIF ( cat .EQ. cat_constant ) THEN
	   CALL IS_CONSTANT( memory, *2000, status )
	   GOTO 5000
* ... get a counter variable
	ELSEIF ( cat .EQ. cat_counter_var ) THEN
	   CALL IS_COUNTER_VAR   (var, memory, *2000, status )
	   GOTO 5000

        ENDIF

* 3/99 complete missing context limits as full-span: to improve cache hits
	CALL COMPLETE_MISSING_LIMITS( cx )

* check in memory to see if we already have the data 
	CALL FIND_MEM_VAR( cx, mr, search_by_value )
	IF ( mr .GT. mr_nonexist ) THEN
* ... yes - already in memory
	   is_mr( isp ) = mr
	   IF ( mr_category(mr) .EQ. cat_user_var ) THEN
	      CALL PASS_AMBIG_LIMS( cx, mr, status )
	      IF ( status .NE. ferr_ok ) GOTO 5000
	   ENDIF
	   GOTO 2000						! pop stack
	ENDIF

******************************************************************************
* * * * * * * * STRIP AWAY TRANSFORMATIONS ONTO STACK * * * * * * * * * * * * 
******************************************************************************

* ... strip off a transformation ?
* ... reverse loop order ensures that X transformations are done BEFORE
* ... the Y transformations. This is necessary for accuracy wrt. COS factors.
* ... By stripping X last it will be done first.
* 10/98 *jc* - need to ignore trans_reduce (see info in ctx/add_rdc_xform.F)
	DO 200 idim = 4, 1, -1
       IF ( cx_trans( idim, cx ) .NE. trans_no_transform
     .          .AND. cx_trans( idim, cx ) .NE. trans_reduced ) THEN
	      CALL IS_STRIP( isact_class_trans, idim, *10, status )
	      GOTO 5000
	   ENDIF
 200	CONTINUE

* ... strip off "modulo" replications
*   ... but do not strip modulo at the topmost level
*     - else LIST/X=340:380 mod_var[X=@ave] will be wrong
	IF ( cat .EQ. cat_user_var ) THEN
	   IF (uvar_name_code(var)(1:3) .EQ. 'EX#'    ) GOTO 260
	ENDIF	   
	CALL GET_CX_MODULO_AXES(cx,modulo_axes)
	DO 250 idim = 1, 4
	   IF ( modulo_axes(idim)
     .	  .AND. cx_lo_ss(cx,idim) .NE. unspecified_int4 ) THEN
	      IF ( cat .EQ. cat_user_var ) THEN			! added 12/95
*    ... cannot modulo an average (or other compressed result)
	         IF ( uvar_given(idim,var) .EQ. uvlim_gvn_xact
     .		 .OR. uvar_given(idim,var) .EQ. uvlim_need_xact ) GOTO 250
*    ... do not apply modulo to pseudo-variables - else their values will not
*	 match the modulo axis indices and values
* 	 Exception to this is modulo regrids -- comments at end of IS_STRIP
	         IF ( uvar_mod_pseudo(idim,var)
     .		.AND. .NOT.ITSA_MODULO_REGRID(cx,idim) ) THEN
*    ... (2/97) imposed axis of a grid changing var can get moduloed 
		   IF ( ITSA_GCVAR(var) ) THEN
	              IF (.NOT.ITSA_MODULO_GCF(cx,idim)) GOTO 250
	           ELSE
	              GOTO 250
	           ENDIF
	         ENDIF
	      ENDIF
! 11/95 - known bug/feature:
! A[I=1:10] behaves inconsistently wrt  A[I=1:20] for A=VAR[I=3:7] on a
! modulo axis of length 10 because is_modulo reconciles differences between
! the component and the result differently from non-modulo vars
	      IF (cx_hi_ss(cx,idim ) .GT. CAXIS_MODLEN(idim,cx)
     .	     .OR. cx_lo_ss(cx,idim ) .LT. 1 ) THEN
	         CALL IS_STRIP( isact_class_modulo, idim, *10, status )
	         GOTO 5000
	      ENDIF
	   ENDIF
 250	CONTINUE

* ... strip off creation of virtual "void" point on subspan modulo axes
	DO 255 idim = 1, 4
	   IF ( modulo_axes(idim)
     .	  .AND. cx_lo_ss(cx,idim) .NE. unspecified_int4 ) THEN
	     IF (TM_ITS_SUBSPAN_MODULO(CGRID_AXIS(idim, cx))) THEN
	      IF (cx_hi_ss(cx,idim) .EQ. CAXIS_MODLEN(idim, cx)) THEN
	        CALL IS_STRIP(isact_class_subspan_fill,idim,*10,status)
	        GOTO 5000
	      ENDIF
	    ENDIF
	  ENDIF
 255	CONTINUE

* ... strip off need for regridding data ?
 260	IF ( cx_unstand_grid( cx ) ) THEN
	   CALL IS_STRIP( isact_class_regrid, no_dim, *10, status )
	   GOTO 5000
	ENDIF

* ... reduce algebraic expression to its components ?
	IF ( cat .EQ. cat_user_var	) THEN
	   CALL IS_ALGEBRA( memory, *10, *2000, isp_base, status )
	   GOTO 5000

* ... read from disk ?
	ELSEIF ( cat .EQ. cat_file_var
     .	.OR.     cat .EQ. cat_file_rev	) THEN

	   IF ( cat .EQ. cat_file_var ) THEN

* ... trim context limits to what is available in file (ds_grid_start/end)
	      CALL RESTRICT_FILE_LIMITS( cx, look_again, status )
	      IF ( status .NE. ferr_ok ) GOTO 5000

* ... kludge to avoid re-reading when regions aren't fully specified 2/90
	      DO 300 idim = 1, 4
	         IF ( cx_lo_ss( cx, idim ) .EQ. unspecified_int4 ) THEN
		    IF ( CGRID_AXIS(idim,cx) .EQ. mnormal ) GOTO 300
	            CALL FLESH_OUT_AXIS( idim, cx, status )
	            IF ( status .NE. ferr_ok ) GOTO 5000
	      	    look_again = .TRUE.
	         ENDIF
 300	      CONTINUE
	      IF ( look_again ) THEN
* ...         check again in memory - maybe we already have it after all
	         CALL FIND_MEM_VAR( cx, mr, search_by_value )
	         IF ( mr .GT. mr_nonexist ) THEN
* ...            yes - already in memory ... worth the trouble
	            is_mr( isp ) = mr
	            GOTO 2000					! pop stack
	         ENDIF
	      ENDIF
	   
* ... check to see if any axes need reversal
	      IF ( TM_AXES_REVERSED(var,rev_axes) .GT. 0 ) THEN
	         CALL IS_STRIP( isact_class_reverse, 1, *10, status )
	         GOTO 5000
	      ENDIF
	   ENDIF

	   CALL IS_READ( memory, *10, *2000, status )
	   GOTO 5000

* ... calculate from internal model-diagnostic code ?
	ELSEIF ( cat .EQ. cat_calc_var	) THEN
	   CALL IS_CALC   ( memory, *10, *2000, status )
	   GOTO 5000

* ... get a "constant variable" (say, "{1,3,5}")
	ELSEIF ( cat .EQ. cat_const_var	) THEN
	   CALL IS_CONST_VAR   ( memory, *10, *2000, status )
	   GOTO 5000

* ... get a string in qouble quotes
	ELSEIF ( cat .EQ. cat_string	) THEN
	   CALL IS_STRING   ( memory, *10, *2000, status )
	   GOTO 5000

* ... get an attribute value
	ELSEIF ( cat .EQ. cat_attrib_val ) THEN
	   CALL IS_ATTRIB_VAL   ( memory, *10, *2000, status )
	   GOTO 5000

* ... get a counter variable
c	ELSEIF ( cat .EQ. cat_counter_var ) THEN
c	   CALL IS_COUNTER_VAR   ( var, memory, *10, *2000, status )
c	   GOTO 5000

	ELSE
	   STOP 'is_unkn_object'
	ENDIF

******************************************************************************
* * * * * * POP THE STACK - A REQUESTED GRID HAS JUST BEEN OBTAINED * * * * *
******************************************************************************
* a ready-to-use-component has been found/created/read/etc.
* it will now be flagged as a component of a higher level operation
* if it has just been created it is not on the delete chain so just set
* its protection with one claim of "in-use".
* (9/99) when evaluating the SAMPLEI function, the intermediate result
* returned here might be a temporary variable. If so, leave it alone
 2000	mr = is_mr( isp )
	IF ( mr_protected(mr) .EQ. mr_temporary ) THEN
	   CONTINUE	! 9/99 - container result for SAMPLE* function
	ELSEIF ( mr_protected(mr) .EQ. mr_in_progress ) THEN
	   mr_protected(mr) = mr_not_protected + 1
	ELSEIF ( mr_protected(mr) .NE. mr_perm_protected ) THEN
	   CALL MR_IN_USE( mr )
	ENDIF

* check for asychronous interrupt received
	IF (interrupted) CALL ERRMSG( ferr_interrupt,status,' ',*5000 )


* DONE ?
 2120	IF     ( isp .LT. isp_base ) THEN
* ... yes - but with a serious error
	   STOP 'isp_undfl'	! IS stack underflowed
	ELSEIF ( isp .EQ. isp_base ) THEN
* ... yes - clean up and go home
	   CALL RELEASE_WORK_SPC
	   RETURN
	ELSE
* ... no - go back for more
*    ... since the stack activity at this level is now completed it contains a
*	 component that is needed by the next lower level.  Change its activity
*	 to reflect its belonging to the lower level (needed when an algebraic
*	 operation needs multiple components so at each component isp level
*	 the "action" is still algebra after the component is obtained)
	   act = is_act( isp - 1 )
	   is_act( isp ) = act
	   isp = isp - 1

******************************************************************************
* * * * * PERFORM TRANSFORMATIONS, ARITHMETIC, RE-GRIDDING, ETC. * * * * * *
******************************************************************************
	   IF     ( act .EQ. isact_request		) THEN
	      GOTO 10					! grid just obtained
	   ELSEIF ( act .EQ. isact_reconcile		) THEN
	      CALL IS_RECONCILE( memory, *10, *2500, status ) ! 11/95
	   ELSEIF ( act .EQ. isact_calculate		) THEN
	      CALL IS_CALC   ( memory, *10, *2500, status ) ! temp ?? needed ?
	   ELSEIF ( act .EQ. isact_algebra		) THEN
	      CALL IS_ALGEBRA( memory, *10, *2500, isp_base, status )
	   ELSEIF ( act .EQ. isact_modulo		) THEN
	      CALL IS_MODULO( memory, *10, *2500, status )     ! 12/94
	   ELSEIF ( act .EQ. isact_subspan_fill		) THEN
	      CALL IS_SUBSPAN_FILL( memory, *10, *2500, status )! 7/02
	   ELSEIF ( act .EQ. isact_reverse		) THEN
	      CALL IS_REVERSE( memory, *10, *2500, status )     ! 2/95
	   ELSEIF ( act .EQ. isact_gather		) THEN
	      CALL IS_GATHER( memory, *10, *2500, status )
	   ELSEIF ( act .EQ. isact_sample		) THEN
	      CALL IS_SAMPLE( memory, *10, *2500, status )
	   ELSEIF ( act .EQ. isact_interp		) THEN
	      CALL IS_TRANS( memory, DO_INTERPOLATE, *2500, status )
	   ELSEIF ( act .EQ. isact_average		) THEN
	      CALL IS_TRANS( memory, DO_AVERAGE, *2500, status )
	   ELSEIF ( act .EQ. isact_4d_ave		) THEN	! 7/96
	      CALL IS_TRANS( memory, DO_4D_AVE, *2500, status )
	   ELSEIF ( act .EQ. isact_integ_def		) THEN	! 7/96
	      CALL IS_TRANS( memory, DO_INTEG_DEF, *2500, status )
	   ELSEIF ( act .EQ. isact_4d_int_def		) THEN
	      CALL IS_TRANS( memory, DO_4D_INT_DEF, *2500, status )
	   ELSEIF ( act .EQ. isact_shift		) THEN
	      CALL IS_TRANS( memory, DO_SHIFT, *2500, status )
	   ELSEIF ( act .EQ. isact_smth_box		) THEN
	      CALL IS_TRANS( memory, DO_SMTH_BOX, *2500, status )
	   ELSEIF ( act .EQ. isact_smth_binml		) THEN
	      CALL IS_TRANS( memory, DO_SMTH_BINML, *2500, status )
	   ELSEIF ( act .EQ. isact_smth_hanng		) THEN
	      CALL IS_TRANS( memory, DO_SMTH_HANNG, *2500, status )
	   ELSEIF ( act .EQ. isact_smth_parzn		) THEN
	      CALL IS_TRANS( memory, DO_SMTH_PARZN, *2500, status )
	   ELSEIF ( act .EQ. isact_smth_welch		) THEN
	      CALL IS_TRANS( memory, DO_SMTH_WELCH, *2500, status )
	   ELSEIF ( act .EQ. isact_fill_ave		) THEN
	      CALL IS_TRANS( memory, DO_FILL_AVE, *2500, status )
	   ELSEIF ( act .EQ. isact_fill_interp		) THEN
	      CALL IS_TRANS( memory, DO_FILL_INTERP, *2500, status )
	   ELSEIF ( act .EQ. isact_fill_near		) THEN
	      CALL IS_TRANS( memory, DO_FILL_NEAR, *2500, status )
	   ELSEIF ( act .EQ. isact_deriv_fwrd		) THEN
	      CALL IS_TRANS( memory, DO_DERIV_FRWD, *2500, status )
	   ELSEIF ( act .EQ. isact_deriv_cntr		) THEN
	      CALL IS_TRANS( memory, DO_DERIV_CNTR, *2500, status )
	   ELSEIF ( act .EQ. isact_deriv_bkwd		) THEN
	      CALL IS_TRANS( memory, DO_DERIV_BKWD, *2500, status )
	   ELSEIF ( act .EQ. isact_min			) THEN
	      CALL IS_TRANS( memory, DO_MIN, *2500, status )
	   ELSEIF ( act .EQ. isact_max			) THEN
	      CALL IS_TRANS( memory, DO_MAX, *2500, status )
	   ELSEIF ( act .EQ. isact_integ_indef		) THEN
	      CALL IS_TRANS( memory, DO_INT_INDEF, *2500, status )
	   ELSEIF ( act .EQ. isact_locate		) THEN
	      CALL IS_TRANS( memory, DO_LOCATE, *2500, status )
	   ELSEIF ( act .EQ. isact_sum  		) THEN
	      CALL IS_TRANS( memory, DO_SUM, *2500, status )
	   ELSEIF ( act .EQ. isact_run_sum		) THEN
	      CALL IS_TRANS( memory, DO_RUN_SUM, *2500, status )
	   ELSEIF ( act .EQ. isact_variance		) THEN
	      CALL IS_TRANS( memory, DO_VARIANCE, *2500, status )
	   ELSEIF ( act .EQ. isact_good_pt		) THEN
	      CALL IS_TRANS( memory, DO_NGOOD, *2500, status )
	   ELSEIF ( act .EQ. isact_bad_pt		) THEN
	      CALL IS_TRANS( memory, DO_NBAD, *2500, status )
	   ELSEIF ( act .EQ. isact_wgted_eq		) THEN
	      CALL IS_TRANS( memory, DO_DIST_LOC, *2500, status )
	   ELSEIF ( act .EQ. isact_regrid		) THEN
	      CALL IS_REGRID( memory, *2500, status )
	   ELSEIF ( act .EQ. isact_autogrid		) THEN
	      CALL IS_AUTOGRID( *2500, status )
	   ELSEIF ( act .EQ. isact_4d_var		) THEN	! 7/96
	      CALL IS_TRANS( memory, DO_4D_VAR, *2500, status )
	   ELSEIF ( act .EQ. isact_4d_good_pt		) THEN	! 7/96
	      CALL IS_TRANS( memory, DO_4D_NGD, *2500, status )
	   ELSEIF ( act .EQ. isact_4d_bad_pt		) THEN	! 7/96
	      CALL IS_TRANS( memory, DO_4D_NBD, *2500, status )
	   ELSEIF ( act .EQ. isact_nrst_dist_abv	) THEN	! 9/97
	      CALL IS_TRANS( memory, DO_NEAREST_DIST_ABOVE, *2500, status )
	   ELSEIF ( act .EQ. isact_nrst_dist_blw	) THEN	! 9/97
	      CALL IS_TRANS( memory, DO_NEAREST_DIST_BELOW, *2500, status )
	   ELSEIF ( act .EQ. isact_nrst_indx_abv	) THEN	! 9/97
	      CALL IS_TRANS( memory, DO_NEAREST_INDEX_ABOVE, *2500, status )
	   ELSEIF ( act .EQ. isact_nrst_indx_blw	) THEN	! 9/97
	      CALL IS_TRANS( memory, DO_NEAREST_INDEX_BELOW, *2500, status )
	   ELSEIF ( act .EQ. isact_event_mask		) THEN	! 6/01
	      CALL IS_TRANS( memory, DO_EVENT, *2500, status )
	   ELSEIF ( act .EQ. isact_read			) THEN
	      WRITE (6,*) 'Tell FERRET programmer to check INTERP_STACK'
* ...	      could set "cat" and branch to above IS_READ
	      CALL IS_READ( memory, *10, *2000, status )    ! temp ?? needed ?
	   ELSE
              CALL ERRMSG( ferr_internal, status,
     .             'unknown transform', *5000 )
	   ENDIF
	   GOTO 5000		! error performing operation
	ENDIF

* 11/95 detour
* A result has just been obtained from a component.
* If the component was non-reusable then so is the result
* NOTE: this is not a complete check for inheritance of non-reusable results!!
*	- multiple components may have been used (e.g. IS_DO_OP)
*	- multiple results may have been computed (e.g. FFT's)
* Those special cases must be tracked inside the computing routines 
 2500	mr     = is_mr( isp )	! result (output)
	com_mr = is_mr( isp+1)	! component (input)
* ...  even if com_mr is wrong it doesn't cause trouble
	IF ( ITS_UN_CACHED( com_mr ) ) CALL UN_CACHE( mr )
	GOTO 2000

* error exit - clean up
 5000	CALL RELEASE_WORK_SPC
	CALL DELETE_MRS_IN_PROGRESS
	CALL UNPROTECT_ALL_MRS
	CALL DEALLO_ALL_GRIDS
        CALL CLEAN_STACK
	RETURN
	END
