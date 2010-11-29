	SUBROUTINE IS_TRANS( memory, ROUTINE, *, status )

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
* create a complete context describing the result of the transformation
* create a memory resident slot to receive the result and initiate the
* transformation

* programmer - steve hankin
* NOAA/PMEL, Seattle, WA - Tropical Modeling and Analysis Program
* written for VAX computer under VMS operating system
*
* V200:  6/1/89 - (based on IS_PLANE_TRANS)
*	10/16/89 - set up adjustable array limits with NON_ARRAY_SUBSC
*	 1/21/90 - pass ss limits from component back to result if unknown
* Unix/RISC port - 1/91 - need to declare "ROUTINE" external (???)
* V230:   8/2/92 - add comment re: cx_valid
*        9/10/92 - bypass error if interpolate is applied to a point
* V312: 5/94 - array "memory" as a calling argument
* V420: 11/95 - fix bug in explicit limits: LET A=TEMP[I=1:160]; STAT A[I=@MAX]
*		... result has one valid and 159 invalid points		 
* V430: 7/96 - incorporate 4D transformations
* V440: 8/96 - allocate memory for COS(lat) array before calling ROUTINE
* V450: 5/97 - fillers different from smoothers when component data is
*		insufficient for result. Should return original data rather
*		than an all-bad field.
* V455:  9/97 - OFFSET_SS returns status
*	11/13/97 - bug fix: @ITP component WW coords must bracket result
*	        (11/18/97 - repaired bug introduced 11/13)
* V491:  1/2/98 - component size can exceed result in case like
*                       let vx = SIN(X[X=1:60:1])
*                       list/i=13:18 vx[x=@sbx:3]
*               As an easy (but not necessarily "correct") fix  we REDUCE
*		the component size here to fit the result. We need to flag the
*		result as not cache-retreivable because the uvar_given value
*		indicates the result size is self-defined (Is this risky??)
* V500:  1/99 *sh* Trap as errors attempts to apply transform to string
*		variables
* V530:  8/00 *sh* changed string trap to be based on data type, not category
* V533:  7/01 *sh* transform @SHF supported for strings
* v552   4/03 *acm* up VAR_TRANS to 150 characters (variables are up to 128, plus
*                   space for the transformation specifier)
* V600  10/05 *acm* Fis bug 899: rather than the ungraceful STOP, issue an error
*                   message if a 4-D transform was requested but the dimensions 
*                   are not available in the grid.

	include	'ferret.parm'
	include	'errmsg.parm'
	include	'interp_stack.parm'
	include	'xcontext.cmn'
	include	'xvariables.cmn'
	include	'xtext_info.cmn'
	include	'xalgebra.cmn'

* calling argument declarations:
	INTEGER	status
	REAL	memory( mem_blk_size, max_mem_blks )
* normal exit is for error conditions
* exit #1 is to signal completion of the activity at this stack level
	INTEGER	ROUTINE		! (returns status )
	EXTERNAL ROUTINE	! added for Unix/RISC/FORTRAN

* internal variable declarations:
* ... "res"="result" , "com"="component"
	LOGICAL	ITSA_FILL_TRANS, its_4d, string_bad_good, string_com
	INTEGER CX_DIM_LEN, MGRID_SIZE_DELTA, DO_4D_STRING_TRANS,
     .		slen, cx_valid,
     .		res_isp, com_isp, res_cx, com_cx, com_mr, res_mr, idim,
     .		trans, grid, lo_off, hi_off, ss_lo, ss_hi, wksiz, wkblk,
     .		mrlist(2), axes(4), idim1, i, ndim
	REAL	arg
	REAL*8	TM_WORLD, ww, ww_lo, ww_hi
	CHARACTER VAR_TRANS*150, buff*150

* --- end of introductory code ---

* initialize
	res_isp	= isp
	com_isp	= res_isp + 1
	res_cx  = is_cx( res_isp )
	com_cx  = is_cx( com_isp )
	com_mr	= is_mr( com_isp )
	idim1	= is_axis( res_isp )
	trans	= cx_trans( idim1, res_cx )
	arg	= cx_trans_arg( idim1, res_cx )
	grid    = cx_grid( com_cx )
	string_bad_good = .FALSE.
	string_com = .FALSE.

* determine if this is a 1D or a 4D (multiple axis) transformation
	its_4d = .FALSE.
	DO 110 i = 1, num_4d_trans
           IF ( trans .EQ. alg_4d_trans(i) ) its_4d = .TRUE.
 110	CONTINUE

* determine the axes to be transformed.
	IF (its_4d) THEN
	   ndim = 0
	   DO 120 idim = 1, 4
	     IF ( cx_trans(idim,res_cx) .EQ. trans ) THEN
	       ndim = ndim + 1
	       axes(ndim) = idim
* ... odd nesting of transforms that may happen from implicit interpolation
	       IF (cx_trans(idim,com_cx).NE.trans_no_transform) GOTO 5200
	     ENDIF
 120	   CONTINUE
	   IF (ndim.LE.1) GOTO 5400     !!  STOP 'is_trans:ndim = 0'
	ELSE
	   ndim = 1
	   axes(1) = idim1
* ... odd nesting of transforms that may happen from implicit interpolation
	   IF ( cx_trans(idim1,com_cx) .NE. trans_no_transform ) GOTO 5200
	ENDIF

* error check: attempt to transform a single point ?
* ( note: this won't catch a[k=1:5@sbx] where a=u[k=5]
* 	  but using CX_DIM_LEN rules out [Z=10:11@ave] within a single point)
	DO 200 i = 1, ndim
	  idim = axes(i)
	  IF ( cx_lo_ss(com_cx,idim) .EQ. unspecified_int4 ) THEN
             IF ( trans .NE. trans_interpolate ) GOTO 5100
	  ENDIF
 200	CONTINUE

* *sh* 10/92 MODE INTERPOLATE has lead to @ITP applied to inappropriate limits
*  errors were trapped mistakenly here for:
*       LIST/X=5.5  5.0 
*  and  LIST/X=5.5/Y=5.5 X  (err on Y axis)
*  this patch returns the component data as the result
        IF ( trans .EQ. trans_interpolate ) THEN
	   IF ( cx_lo_ss(com_cx,idim1) .EQ. unspecified_int4 ) THEN
              CALL TRANSFER_CONTEXT( com_cx, res_cx )
              CALL CREATE_MEM_VAR( res_cx, res_mr, status )
              IF ( status .NE. ferr_ok ) GOTO 5000
              CALL COPY_GRID(
     .           memory( 1, mr_blk1(com_mr) ) , com_mr,
     .           memory( 1, mr_blk1(res_mr) ) , res_mr )
              is_mr( res_isp ) = res_mr
              GOTO 1000
            ENDIF
         ENDIF

* create a result context based on the received component
	cx_valid= cx_buff
	CALL TRANSFER_CONTEXT( com_cx, cx_valid )
		DO 300 i = 1, ndim
	   idim = axes(i)
* ... if the requested region was unknown use the component region
	   IF ( cx_lo_ss(res_cx,idim).EQ.unspecified_int4
     .	   .AND.cx_lo_ww(idim,res_cx).EQ.unspecified_val8 ) THEN
* 11/95 - FLESH_OUT_AXIS cannot be trusted to flesh this out.
*	  It puts responsibility for redundant limits onto the various interp
*	  stack routines (I guess ...)
	      CALL TRANSFER_AXIS( idim, com_cx, res_cx )
	      cx_trans( idim, res_cx ) = trans
	      cx_trans_arg( idim, res_cx ) = arg
	      cx_by_ss(idim,res_cx) = .TRUE.
	      IF ( trans  .GT. trans_compress_code ) THEN
	         cx_lo_ss( res_cx, idim ) = unspecified_int4
	         cx_hi_ss( res_cx, idim ) = unspecified_int4
	         cx_by_ss( idim, res_cx ) = .FALSE.
	      ENDIF
! pre=11/95cx_lo_ss(res_cx,idim) = cx_lo_ss(com_cx,idim)
!	      cx_hi_ss(res_cx,idim) = cx_hi_ss(com_cx,idim)
!	      cx_by_ss(idim,res_cx) = .TRUE.

* 11/95 - is interpolation the only case of a compressing transform
* where the component ww range normally doesn't match the result?
	   ELSEIF ( trans .EQ. trans_interpolate ) THEN
* ... component range must **bracket** the result, however (11/97)
	      ww    = cx_lo_ww(idim,res_cx)
	      ss_lo = cx_lo_ss(com_cx,idim)
	      ss_hi = cx_hi_ss(com_cx,idim)
	      ww_lo = TM_WORLD( ss_lo, grid, idim, box_lo_lim )
	      ww_hi = TM_WORLD( ss_hi, grid, idim, box_hi_lim )
* ... allow ww_lo>ww_hi to accomodate time encodings
	      IF ( (ww-ww_lo)*(ww-ww_hi) .GT. 0.0D0 ) GOTO 5300 

* 11/95 - compressing transform where variable limits are in a nested def'n
*    e.g.   list b[i=1:5] where b = a[i=@ave] and a = temp[i=1:100]
* ... use the inner limits (i=1:100) for the transformation
	   ELSEIF ( trans .GT. trans_compress_code
     .      .AND. (cx_lo_ww(idim,res_cx).NE.cx_lo_ww(idim,com_cx)
     .	    .OR.cx_hi_ww(idim,res_cx).NE.cx_hi_ww(idim,com_cx))   ) THEN
	      CALL TRANSFER_AXIS( idim, com_cx, res_cx )
	      cx_trans( idim, res_cx ) = trans
	      cx_lo_ss( res_cx, idim ) = unspecified_int4
	      cx_hi_ss( res_cx, idim ) = unspecified_int4
	      cx_by_ss( idim, res_cx ) = .FALSE.
	   ENDIF

	   CALL TRANSFER_AXIS( idim, res_cx, cx_valid )
	   CALL FLESH_OUT_AXIS( idim, cx_valid, status )
	   IF ( status .NE. ferr_ok ) GOTO 5000

 300	CONTINUE

* the context is ready - carve out memory space for it
	CALL TRANSFER_CONTEXT( cx_valid, res_cx )
	
	IF ( cx_type(com_cx) .EQ. ptype_string ) THEN
	   string_com = .TRUE.
	   IF (trans .EQ. trans_good_pt    .OR.
     .         trans .EQ. trans_bad_pt     .OR.
     .         trans .EQ. trans_4d_good_pt .OR.
     .         trans .EQ. trans_4d_bad_pt ) THEN
              string_bad_good = .TRUE.
              cx_type(res_cx) = ptype_float
           ENDIF
        ENDIF

	CALL CREATE_MEM_VAR( res_cx, res_mr, status )
	IF ( status .NE. ferr_ok ) GOTO 5000
	is_mr( res_isp ) = res_mr
	IF ( mr_type(res_mr) .EQ. ptype_string .AND.
     .       (.NOT.string_bad_good)) THEN
* ... only applicable to @SHF transformation; else will return a null string.
           CALL INIT_C_STRING_ARRAY( MGRID_SIZE_DELTA(res_mr),
     .                               memory(1, mr_blk1(res_mr)),
     .                               mr_c_pointer(res_mr)      )
	ENDIF

!* diagnostic message ?
!	IF (mode_diagnostic) CALL DIAGNOSTIC_OUT(
!     .		'@'//ALG_TRANS_CODE(trans)//'   ', res_mr, point_to_mr )

* component size may be inadequate for requested result
* pre-fill result with missing data flag if component is inadequate
* (7/96 - note that 4D transforms are always compressing transforms)
	IF ( trans .LT. trans_compress_code ) THEN
	   idim = idim1				! should be equal already
	   IF ( ITSA_FILL_TRANS(trans) ) THEN	! 5/97 - filler are difrnt
	      lo_off = 0
	      hi_off = 0
* 5/97: for fillers the DO_ routines handle result regions beyond the
*	bounds of the component data 
	   ELSE
	      CALL OFFSET_SS( idim, res_cx, lo_off, hi_off, status )
	      IF ( status .NE. ferr_ok ) GOTO 5000
	   ENDIF
* ... computable limits from component
* Note (8/92): when the requested limites are unknown the com_cx will be
*              full-width along idim so subtracting the offset limits leaves
*              cx_valid as beyond the possible range but no known crash results
	   cx_lo_ss( cx_valid, idim ) = cx_lo_ss( com_cx, idim ) - lo_off
	   cx_hi_ss( cx_valid, idim ) = cx_hi_ss( com_cx, idim ) - hi_off
	   IF ( cx_lo_ss(cx_valid,idim) .GT. cx_hi_ss(cx_valid,idim) ) THEN
* ... insufficient component data to compute anything
	      IF ( mr_type(res_mr) .EQ. ptype_string ) THEN
	        CALL SET_NULL_C_STRING_ARRAY(mr_c_pointer(res_mr),
     .                                       MGRID_SIZE_DELTA(res_mr) )
	      ELSE
	        CALL FILL_MEMORY( memory,mr_blk1(res_mr),mr_nblks(res_mr),
     .		  		  mr_bad_data(res_mr) )
	      ENDIF
	      GOTO 1000
	   ELSEIF (  cx_lo_ss(res_cx,idim) .LT. cx_lo_ss(cx_valid,idim)
     .		.OR. cx_hi_ss(res_cx,idim) .GT. cx_hi_ss(cx_valid,idim) ) THEN
* ... insufficient component data to compute the entire requested region
	      CALL BAD_EDGES( cx_valid, memory(1,mr_blk1(res_mr)), res_mr )
	   ELSEIF (  cx_lo_ss(res_cx,idim) .GT. cx_lo_ss(cx_valid,idim)
     .		.OR. cx_hi_ss(res_cx,idim) .LT. cx_hi_ss(cx_valid,idim) ) THEN
* ... *1/98* component data is BIGGER than expected for requested result
* ... reduce the apparent size of the component data (dangerous??)
* e.g.                      let vx = SIN(X[X=1:60:1])
*                           list/i=13:18 vx[x=@sbx:3]
*     Note that this situation occurs when the component variable being xformed
*	thinks that it has self-describing limits (uvar_given<uvlim_gvn_xact).
*       Since the transformed version of the variable (the result) is getting
*	its limits from elsewhere uncache the result so it won't be recalled.
*	==> THIS IS A KLUDGE: The right solution is that the limits i=13:18
*	should never get applied to vx ... but this is too difficult to
*	implement in GET_VAR_CONTEXT and PARSE_VAR_NAME
	      CALL UN_CACHE( res_mr )
* ... fillers and smoothers are equivalent here ... so recompute offsets
	      CALL OFFSET_SS( idim, res_cx, lo_off, hi_off, status )
	      IF ( status .NE. ferr_ok ) GOTO 5000
	      IF ( cx_lo_ss(res_cx,idim) .GT. cx_lo_ss(cx_valid,idim) )
     .			cx_lo_ss(com_cx,idim) = cx_lo_ss(res_cx,idim) + lo_off
	      IF ( cx_hi_ss(res_cx,idim) .LT. cx_hi_ss(cx_valid,idim) )
     .			cx_hi_ss(com_cx,idim) = cx_hi_ss(res_cx,idim) + hi_off
	      cx_by_ss(idim,res_cx) = .TRUE.
	      CALL FLESH_OUT_AXIS( idim, com_cx, status )
	      IF ( status .NE. ferr_ok ) GOTO 5000
	   ENDIF
	ENDIF

* set up 4D array limits as unsubscripted variables in COMMON
	mrlist(1) = com_mr
	mrlist(2) = res_mr
	CALL NON_ARRAY_SUBSC( mrlist, 2 )

* transformations applied to string arguments
	IF ( string_com ) THEN
	   IF (trans .EQ. trans_shift) THEN
	      CALL DO_STRING_SHIFT ( idim, arg,
     .		memory( 1, mr_blk1(com_mr) ), com_mr, com_cx, ! component
     .		memory( 1, mr_blk1(res_mr) ), res_mr, res_cx) ! result
	   ELSEIF (string_bad_good) THEN
	   
	      IF (its_4d) THEN
	         status = DO_4D_STRING_TRANS( trans,
     .		 memory( 1, mr_blk1(com_mr) ), com_mr, com_cx, ! component
     .		 memory( 1, mr_blk1(res_mr) ), res_mr, res_cx) ! result
	         IF ( status .NE. ferr_ok ) GOTO 5000
              ELSE
 	         CALL DO_STRING_GOODBAD ( idim, trans,
     .		 memory( 1, mr_blk1(com_mr) ), com_mr, com_cx, ! component
     .		 memory( 1, mr_blk1(res_mr) ), res_mr, res_cx) ! result                            ! result
	      ENDIF
	   ELSE
	     CALL SET_NULL_C_STRING_ARRAY(mr_c_pointer(res_mr),
     .                                    MGRID_SIZE_DELTA(res_mr) )
	   ENDIF
	   GOTO 1000
	ENDIF

* reserve working space for the transformation
* Note: this logic is a loose hodge-podge to get an upper bound size for all
* the different cases: 4D, 1D, with arguments, etc.
	wksiz = 1
	DO 500 i = 1, ndim
	   idim = axes(i)
	   wksiz = wksiz + CX_DIM_LEN(idim,com_cx)
 500	CONTINUE
* ... allow room for COS(lat) array during 4D transforms (8/96)
	IF (ndim .GT. 1 )  wksiz = wksiz + CX_DIM_LEN(y_dim,com_cx)
* ... set minimum work size
	wksiz = MAX( iswksiz, wksiz )
* ... for transforms like smoothers allow extra slop on each end 
	IF ( arg .NE. bad_val4 )  wksiz =
     .			MAX( wksiz, CX_DIM_LEN(idim1,res_cx)+INT(arg) )
	CALL GET_WORK_SPC( wksiz, wkblk, status )
	IF ( status .NE. ferr_ok ) GOTO 5000

* do the transform

	status = ROUTINE( idim, arg,
     .		memory( 1, mr_blk1(com_mr) ), com_mr, com_cx, ! component
     .		memory( 1, mr_blk1(res_mr) ), res_mr, res_cx, ! result
     .		memory( 1, wkblk )			    ) ! work space
	IF ( status .NE. ferr_ok ) GOTO 5000

* remove component protections that signaled this calculation needed it
 1000	CALL MR_NOT_IN_USE( com_mr )	

* free the context stack space
	cx_stack_ptr = cx_stack_ptr - 1

* success
	RETURN 1

* error exit
 5000	RETURN
 5100	buff = VAR_TRANS( idim, res_cx, slen )
	CALL ERRMSG( ferr_trans_nest, status,
     .		     buff(:slen)//' applied to point', *5000 )
 5200	buff = VAR_TRANS( idim, com_cx, slen )
	CALL ERRMSG( ferr_trans_nest, status, 'interpolation on '//
     .		     buff(:slen)//'--> DEFINE a new variable', *5000 )
 5300	buff = VAR_TRANS( idim, res_cx, slen )
	CALL ERRMSG( ferr_limits, status,
     .		     buff(:slen)//' exceeds interpolation range',
     .		     *5000 )
 5400	CALL ERRMSG( ferr_trans_nest, status,
     .		     ' not all dimensions present on grid',
     .		     *5000 )

	END	
