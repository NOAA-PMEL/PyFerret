	SUBROUTINE FIND_MEM_VAR( cx, mr, by_value )

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

#ifdef unix
	include 'tmap_format/tmap_dims.parm'
	include 'tmap_format/xtm_grid.cmn_text'
	external xgt_grid_data
	include 'ferret_cmn/ferret.parm'
	include 'ferret_cmn/errmsg.parm'
	include 'ferret_cmn/xvariables.cmn'
	include 'ferret_cmn/xcontext.cmn'
	include 'ferret_cmn/xprog_state.cmn'
#else
	INCLUDE 'TMAP_FORMAT:TMAP_DIMS.PARM'
	INCLUDE 'TMAP_FORMAT:XTM_GRID.CMN'
	INCLUDE 'FERRET_CMN:FERRET.PARM'
	INCLUDE 'FERRET_CMN:ERRMSG.PARM'
	INCLUDE 'FERRET_CMN:XVARIABLES.CMN'
	INCLUDE 'FERRET_CMN:XCONTEXT.CMN'
	INCLUDE 'FERRET_CMN:XPROG_STATE.CMN'
#endif

* calling argument declarations:
	INTEGER		cx, mr
	LOGICAL		by_value

* internal variable declarations:
	LOGICAL		has_trans, itsa_uvar, normal(4), non_modulo(4)
	INTEGER		HASH_CX, line,
     .			mv, defining_grid, idim, new_mr, var, status
        REAL*8          TM_WORLD, ww_hi_bound

* in "STUPID" mode the program can't find anything in memory
	IF ( mode_stupid ) GOTO 5000

* initialize
	defining_grid = cx_grid( cx )
	var	      = cx_variable(cx)

* is there a completed context to look for ?
	IF ( defining_grid   .EQ. unspecified_int4
     .	.OR. cx_data_set(cx) .EQ. unspecified_int4
     .	.OR. var             .EQ. unspecified_int4 ) GOTO 5000

* cat_const_vars are never cached (see IS_CONST_VAR)
* ditto for cat_string (as-of 1/99)
	IF ( cx_category(cx) .EQ. cat_const_var
     .  .OR. cx_category(cx) .EQ. cat_string   ) GOTO 5000

* are we looking for  user defined variable ?
	itsa_uvar = cx_category(cx) .EQ. cat_user_var

* do we need special checks for transforms ?
	has_trans = .FALSE.
	DO 200 idim = 1, 4
 200    has_trans = has_trans
     .        .OR. (cx_trans(idim,cx).NE.trans_no_transform)

* flag the normal axes so we can bypass checks on them quickly
	DO 210 idim = 1, 4
	   line = grid_line(idim,defining_grid)
	   IF (line .EQ. mnormal) THEN
	      normal(idim) = .TRUE.
	   ELSE 
	      normal(idim) = .FALSE.
	      non_modulo(idim) = .NOT.line_modulo(line)
	   ENDIF
 210	CONTINUE

* determine memory hash pile to search
	mv = HASH_CX( cx )

* locate next table slot in linked list
 100	mv = mv_flink( mv )
	IF ( mv .LT. 0 ) THEN
	   mr = mr_nonexist
	   RETURN
	ENDIF

* right variable ?
	IF ( mr_variable ( mv ) .NE. cx_variable(cx)
     .	.OR. mr_category ( mv ) .NE. cx_category(cx)
     .	.OR. mr_data_set ( mv ) .NE. cx_data_set(cx)
     .	.OR. mr_grid ( mv ) .NE. defining_grid	     )		GOTO 100

* right instance (coordinate ranges and transformations) ?
	IF ( itsa_uvar ) THEN

* * * * * LOOK FOR A USER-DEFINED VARIABLE BASED ON COORDINATE VALUES  * * * * 
	   DO 40 idim = 1,4

	      IF ( normal(idim) ) GOTO 40

* transformation correct ?
	      IF ( mr_trans(idim,mv) .NE. cx_trans(idim,cx) )	GOTO 100

* limits specified in uvar expression ? -->  no match necessary
* 11/95 - since modulo isn't stripped until after FIND_MEM_VAR is called
*	  we cannot bypass this on modulo axes.
*		e.g. x1 has I=1:10; let xgap=x1[i=3:7]
*		then list xgap[i=1:15] should not "find" the i=3:7 result
	      IF ( uvar_given(idim,var) .LE. uvlim_gvn_xact 
     .		   .AND. non_modulo(idim) 		   ) GOTO 40

* insist that the  grid stored in memory encompasses the requested data
	      IF ( mr_lo_ww(idim,mv) .GT. cx_lo_ww(idim,cx)
     .        .OR. mr_hi_ww(idim,mv) .LT. cx_hi_ww(idim,cx)) GOTO 100

 40	   CONTINUE

* right instance (coordinate ranges and transformations) ?
	ELSEIF ( by_value ) THEN

* * * * * * LOOK FOR A VARIABLE BASED ON COORDINATE VALUES * * * * * * * * 
	   DO 50 idim = 1,4

	      IF ( normal(idim) ) GOTO 50

* transformation correct ?
	      IF ( mr_trans(idim,mv) .NE. cx_trans(idim,cx) )	GOTO 100

* insist that the  grid stored in memory encompasses the requested data
	      IF ( mr_lo_ww(idim, mv) .GT. cx_lo_ww(idim, cx)
     .        .OR. mr_hi_ww(idim, mv) .LT. cx_hi_ww(idim, cx))	GOTO 100

 50	   CONTINUE

	ELSE		! by subscript

* * * * * * LOOK FOR A VARIABLE BASED ON SUBSCRIPT VALUES * * * * * * * * *
	   DO 60 idim = 1,4

	      IF ( normal(idim) ) GOTO 60

* transformation correct ?
	      IF ( mr_trans(idim,mv) .NE. trans_no_transform )	GOTO 100
	      IF ( cx_trans(idim,cx) .NE. trans_no_transform ) STOP 
     .				'find_by_ss_trans'	! temp diagnostic

* insist that the  grid stored in memory encompasses the requested data
* ... this is an adequate test because the grids looked for by subscript are
* ... always untransformed --> normal axes always have a span on 1 point
	      IF ( mr_lo_ss(mv, idim) .GT. cx_lo_ss(cx, idim)
     .	      .OR. mr_hi_ss(mv, idim) .LT. cx_hi_ss(cx, idim))	GOTO 100

 60	   CONTINUE
	ENDIF

* make sure it is completed
	IF ( mr_protected( mv ) .EQ. mr_in_progress )		GOTO 100

* special tests left for last to keep the main loop simple and fast
* ... special limit requirements for indefinite integrals:
*     to satisfy the search an indefinite integral must have exactly the same
*     lower limit and must have upper limit either identical or the grid
*     we are looking for must have upper endpoint on a grid box boundary
*     ( to avoid errors of partial grid boxes )
* fixed logic bug: 8/92 using ww_hi_bound
* was:            IF ( mr_hi_ww(idim,mv).NE.cx_hi_ww(idim,cx)
*     .          .AND. mr_hi_ss(mv,idim).EQ.cx_hi_ss(cx,idim) )  GOTO 100
	IF ( has_trans ) THEN
	   DO 70 idim = 1, 4
	      IF ( normal(idim) ) GOTO 70
	      IF ( cx_trans( idim, cx ) .EQ. trans_integ_indef ) THEN
	         IF ( mr_lo_ww(idim,mv).NE.cx_lo_ww(idim,cx) )	GOTO 100
                 IF ( mr_hi_ww(idim,mv).NE.cx_hi_ww(idim,cx) ) THEN
                    ww_hi_bound = TM_WORLD( cx_hi_ss(cx,idim),
     .                         defining_grid, idim, box_hi_lim ) 
                    IF ( ww_hi_bound .NE. cx_hi_ww(idim,cx)) GOTO 100
                 ENDIF
	      ENDIF
 70	   CONTINUE
	ENDIF

	IF ( has_trans ) THEN
	   DO 80 idim = 1, 4
	      IF ( normal(idim) ) GOTO 80
* ... argument of transform must be right
* ... and transforms that compress to a point must have exact limits
	      IF ( cx_trans_arg(idim,cx).NE.mr_trans_arg(idim,mv) ) GOTO 100
	      IF ( cx_trans(idim,cx) .GT. trans_compress_code ) THEN
	         IF ( mr_lo_ww(idim,mv).NE.cx_lo_ww(idim,cx) )	GOTO 100
	         IF ( mr_hi_ww(idim,mv).NE.cx_hi_ww(idim,cx) )	GOTO 100
	      ENDIF
* ... @RSUM must match lower subscript exactly (1/92)
	      IF ( cx_trans( idim, cx ) .EQ. trans_run_sum ) THEN
                 IF ( mr_lo_ss(mv,idim).NE.cx_lo_ss(cx,idim) )  GOTO 100
              ENDIF
* ... @WEQ must match both hi and lo subscripts (1/92)
	      IF ( cx_trans( idim, cx ) .EQ. trans_wgted_eq ) THEN
                 IF ( mr_lo_ss(mv,idim).NE.cx_lo_ss(cx,idim)
     .           .OR. mr_hi_ss(mv,idim).NE.cx_hi_ss(cx,idim) )  GOTO 100
              ENDIF

 80	   CONTINUE
	ENDIF

* ... if regridded then regridding transforms must be right
	IF ( cx_unstand_grid( cx ) ) THEN
	   DO 90 idim = 1, 4
	      IF ( normal(idim) ) GOTO 90
	      IF ( cx_regrid_trans( idim, cx ) .NE.
     .		   mr_regrid_trans( idim, mv   )		) GOTO 100
 90	   CONTINUE
	ENDIF

* ... if user-defined then watch out for special limit match requirements
	IF ( itsa_uvar ) THEN
	   DO 92 idim = 1, 4
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
	IF (mode_diagnostic) CALL DIAGNOSTIC_OUT('found  ',mr,point_to_mr)

* ... send back pointer
	RETURN

* failure exit
 5000	mr = mr_nonexist
	RETURN
	END
