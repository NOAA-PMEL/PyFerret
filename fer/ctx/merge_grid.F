	SUBROUTINE MERGE_GRID(	iarg, its_gc, com_text,
     .				mrg_cx, com_cx, uvar, status )

* combine the grid from a new component into the grid information from the
* already-merged components
* compute: 1) the aggregate grid (as a collection of 4 axes)
*	   2) uvar_given - which tells how limits must be specified for each
*		axis of the aggregate grid
*  e.g. DEFINE VAR NEW = U[K=1]+TEMP[K=@AVE] where the 2 variables are on
*					     different axes in all 4 dimensions
*	The aggregate grid will be from U in X,Y and T since it is the first
*	variable in the expression and from TEMP in Z since that axis is neede
*	to label the result properly.
*       UVAR_GIVEN will be "needed" on X,Y and T and "need_exact" on Z since
*	NEW should be re-used from memory-resident storage only when the stored
*	Z limits match the requested exactly.
* note: components are merged in the reverse of the order of computation

* programmer - steve hankin
* NOAA/PMEL, Seattle, WA - Tropical Modeling and Analysis Program
* written for VAX computer under VMS operating system
*
* V200:  6/28/89 - complete re-write
* 	 3/27/90 - re-word warning message a bit
* RISC/Unix port *sh* 11/91: eliminate VMS-specific HELP coaching line
* V230:  1/27/92 - changed uvlim_*_hi (unused) to uvlim_*_lohi to describe
*                  @WEQ - exaxt limits required but a range of data
*                - also added @RSUM (like @IIN)
* V400: 11/9/94	 - use CALL WARN to display "Ambiguous coord"  message
* V420: 9/95	 - Support for implicit grids: deallocate after it is used
*	10/95	 - support for implicit axes: pass along use-count 
*		   responsibility (TM_USE_LINE) for implicit axis to merged
*		   grid since component will get deallocated
*	 1/96	 - use INHERITED_LIMITS to ensure that modulo regridding
*		   limits (embedded inside of expression) are not passed on
*		   e.g. "5 + A[gt=climatology@MOD]" where A=SST[T=lo:hi] 
*	2/27/96	 - axes for which cx_lo_ww was undefined were ignored.
*		   This eliminated consideration of the Y
*		   axis of LAT_ARG in r*lat_arg/LAT_REF	where
*		   LAT_ARG = Y[Y=90S:90N:15]; r = 6; lat_ref = 45
*		   The perverted logic sequence is explained at "2/27"
* V430:	 5/96	 - fixing a bug that seems to have been here all along?????
*		   the logic of this routine is based on cx_given, ignoring
*		   limits given INSIDE of component definitions. Thus 
*			let depth_z=h[x=110w,z=@iin]-h[x=110w]/2
*			let r0=depth_z-x[g=gdepth]
*		   resulted in an "Ambiguous axis" warning, ignoring the X=110W
*		   Added variable expl_com_lims to the code
* V450: 1/97	 - changes to accomodate grid changing functions.
*		   MERGIFY_GRID is no
*		   longer used.  The first component is handled in a manner
*		   symmetrical with respect to the others.
*		   Process a component even though it has unspecified ww lims.
*		   Allow a com and a mrg axis both to be normal (axis_choice).
*		   Maintain cx_given for the merged data set.
*		   Added a row to the ax_choice table to deal with merged
*			axis that was "uvlim_irrelevant"

#ifdef unix
	include 'tmap_format/tmap_dims.parm'
	include	'tmap_format/xtm_grid.cmn_text'
	external xgt_grid_data
	include 'ferret_cmn/ferret.parm'
	include 'ferret_cmn/errmsg.parm'
	include	'ferret_cmn/interp_stack.parm'
	include	'ferret_cmn/xcontext.cmn'
	include	'ferret_cmn/xvariables.cmn'
	include	'ferret_cmn/xprog_state.cmn'
	include	'ferret_cmn/xtext_info.cmn'
	include	'ferret_cmn/xfr_grid.cmn'
#else
	INCLUDE 'TMAP_FORMAT:TMAP_DIMS.PARM'
	INCLUDE	'TMAP_FORMAT:XTM_GRID.CMN'
	INCLUDE 'FERRET_CMN:FERRET.PARM'
	INCLUDE 'FERRET_CMN:ERRMSG.PARM'
	INCLUDE	'FERRET_CMN:INTERP_STACK.PARM'
	INCLUDE	'FERRET_CMN:XCONTEXT.CMN'
	INCLUDE	'FERRET_CMN:XVARIABLES.CMN'
	INCLUDE	'FERRET_CMN:XPROG_STATE.CMN'
	INCLUDE	'FERRET_CMN:XTEXT_INFO.CMN'
	INCLUDE	'FERRET_CMN:XFR_GRID.CMN'
#endif

* calling argument declarations
	LOGICAL		its_gc
	INTEGER		iarg, mrg_cx, com_cx, uvar, status
	CHARACTER	com_text*(*)

* local variable declarations
	LOGICAL		INHERITED_LIMITS, same, nest, expl_com_lim
	INTEGER		CX_DIM_LEN, TM_LENSTR1, REGION_CLASS,
     .			NEST_REGION_CLASS,  WHICH_GCFCN,
     .			mrg_grd, mrg_ax, mrg_len, com_grd, com_ax, com_len,
     .			idim, i, icom, imrg, com_uvar, com_class, mrg_class,
     .			uv_class, ax_choice(6,5), joint_match(0:3,0:3),
     .			lims_gvn, gcfcn, axis_source(4)

* local parameter declarations
	INTEGER		EXP,   IMP,   WRN1,   NP,   NA,  WRN2
	PARAMETER     ( EXP=1, IMP=2, WRN1=3, NP=4, NA=5, WRN2=6 ) 
	INTEGER		gnl,   lo,   lohi,   xact
	PARAMETER     ( gnl=0, lo=1, lohi=2, xact=3 )

* procedure:
* the important considerations are
*	o whether merged/component axes are same axis
*	o whether merged/component axes are "irrelevant"
*	o whether merged/comonent data spans are 1 or >1
*	o whether merged/component region is given explicitly or implicitly
*	o whether merged context depends on default data set

* combinations of explicitly and implicitly defined data will take the axis
* of the implicit component(s)
* e.g. W[K=1:3] + U will have the Z/K axis from component U

* when successfully completed the merged axis will be 
*	1) the axis from the first component with region IMPLICITLY defined
*	      ... or failing that ...
*	2) the axis fom the first component with explicit region >1
*	      ... or failing that ...
*	3) the axis from the first component with any explicit region
* If all components are normal to this axis the result, too, will be normal

* * * * * * * * GRID AXIS SELECTION TABLE * * *
*			  C O M P O N E N T
*	       1      2     3       4     5       6  
*                     cx_given          .NOT.cx_given
*                same          diff      same   diff 
*  size:       1     >1      1     >1
*	    ------------------------------------------
*1       1  | EXP  |  *** | EXP | EXP |  IMP  | IMP  |
*    given  ------------------------------------------
*2      >1  | ***  |  NA  | NA  | WRN1|  IMP  | IMP  |
*           ------------------------------------------
*3.NOT.given| NA   |  NA  | NA  | NA  |  NA   | WRN2 |
*           ------------------------------------------
*4  normal  | ***  |  *** | EXP | EXP |  NA*  | IMP  |   (NA* was *** pre 1/97)
*           ------------------------------------------
*5  irrelev | EXP  |  EXP | EXP | EXP |  IMP  | IMP  |   (new row, 1/97)
*           ------------------------------------------
* EXP  - impose axis and explicit limits from component
* IMP  - impose axis from component with implicit limits
* ***  - impossible to occur
* NA   - not applicable in determining grid (but still relevant in uvar_given)


* * * DETERMINING UVAR_GIVEN * * *
* There are subtlities associated with limits that are given in the UVAR
* definition.  These are classified in uvar_given under these codings:
* ("gvn" indicates lo/hi limits were given in square brackets)
* ("gnl","lo","lohi" and "xact" encode special requirements of transformations)
*	uvlim_irrelevant- limits irrelevant: NEW=10.1
*	uvlim_gvn_gnl	- limits given in a gen'l way: NEW=U[K=1:5]
*	uvlim_gvn_lo	- limits that require exact lo match: NEW=U[K=1:5@IIN]
*	uvlim_gvn_lohi	- limits require exact lohi match: NEW=U[K=1:5@WEQ]
*	uvlim_gvn_xact	- limits that reduce to a pt.: NEW=U[K=1:5@AVE]
*	uvlim_needed	- no limit info. given
*	uvlim_need_lo	- no lims but transf. needs xact lo:NEW=U[K@IIN]
*	uvlim_need_lohi	- no lims but transf. needs xact lohi:NEW=U[K=@WEQ]
*	uvlim_need_xact - no lims but transf. needs xact: NEW=U[K@AVE]
* note: i) uvlim_***_xact requires both limits to match exactly in order to
*	   reuse memory-resident data

* * * * DETERMINING UVAR_GIVEN WITH MULTIPLE COMPONENTS IN EXPRESSION * * *
*		---  aggregate of previous components ---
*			gnl	lo	lohi	  xact
*		------------------------------------------
* N C |	gnl	|	gnl	lo	lohi	gnl/xact |
* E O |	lo	|	 lo	lo	lohi	 lo/xact |
* W M |	lohi	|      lohi	lohi	lohi   lohi/xact |
*   P |	xact	|	gnl	lo	lohi	  xact	 |
*		------------------------------------------
* in this table the rows refer to the context qualifiers of new components
* (i.e. what is given in square brackets).  The result of the table and
* the columns are a generalization of the same concept applied to an aggregate
* of components in an expression
* gnl  == no match required e.g. "U"		or "U[K=1:5]"
* lo   == match lo limit    e.g. "U[K=@IIN]"	or "U[K=1:5@IIN]"
* lohi == match lohi limit  e.g. "U[K=@WEQ]" 
* xact == match both limits e.g. "U[K=@AVE]"	(b case in a/b above)
*	  or neither limit e.g. "U[K=1:5@AVE]"	(a case in a/b above)
* "xact" is special because it refers to transformations that reduce a range
*	of data to a point (like averaging).  If limits are included ("gvn")
* 	it becomes self-documenting and axis-independent.

	DATA	(ax_choice(i,1),i=1,6) / EXP,NP ,EXP,EXP, IMP,IMP  /,
     .		(ax_choice(i,2),i=1,6) / NP ,NA ,NA ,WRN1,IMP,IMP  /,
     .		(ax_choice(i,3),i=1,6) / NA ,NA ,NA ,NA , NA ,WRN2 /,  
     .		(ax_choice(i,4),i=1,6) / NP ,NP ,EXP,EXP, NA ,IMP  /,
     .		(ax_choice(i,5),i=1,6) / EXP,EXP,EXP,EXP, IMP,IMP  /
! pre 1/97 .	(ax_choice(i,4),i=1,6) / NP , NP, EXP, EXP,  NP , IMP  /

	DATA	(joint_match(i,0),i=0,3) /  gnl,   lo, lohi, gnl  /,
     .		(joint_match(i,1),i=0,3) /   lo,   lo, lohi, lo   /,
     .		(joint_match(i,2),i=0,3) / lohi, lohi, lohi, lohi /,
     .		(joint_match(i,3),i=0,3) /  gnl,   lo, lohi, xact /

**************************************************************************
* is this a grid changing variable? (1/97)
	
	IF (its_gc) THEN
	   gcfcn = WHICH_GCFCN(uvar)
	   CALL GCF_GET_AXIS_SRCS(gcfcn, iarg, axis_source)
	ENDIF

* initialize
	mrg_grd = cx_grid( mrg_cx )
	com_grd = cx_grid( com_cx )

* do we have nested user-defined variables ?
	nest = cx_category( com_cx ) .EQ. cat_user_var
	IF ( nest ) com_uvar = cx_variable( com_cx )

* for merged context to be data set independent all components must be so
	IF ( cx_category(com_cx) .EQ. cat_pseudo_var ) THEN
* ... pseudo-vars are merged only if explicit "G=".  Was it G=var or G=grid ?
	   cx_dset_gvn(mrg_cx) = cx_dset_gvn(mrg_cx)
     .				.AND. .NOT.cx_grid_needs_dset
	ELSE
	   cx_dset_gvn(mrg_cx) = cx_dset_gvn(mrg_cx) .AND.
     .	(cx_dset_gvn(com_cx).OR.cx_data_set(com_cx).EQ.pdset_irrelevant)
	ENDIF

* loop through the axes
	DO 500 idim = 1,4

* get axes and lengths of data along them
	   com_ax  = grid_line ( idim, com_grd )
	   mrg_ax  = grid_line ( idim, mrg_grd )
	   com_len = CX_DIM_LEN( idim, com_cx  )
	   mrg_len = CX_DIM_LEN( idim, mrg_cx  )

* arg of grid-changing function?
	IF (its_gc ) THEN
	   IF (    axis_source(idim).EQ.pgc_ignore_axis) THEN
	      GOTO 500
	   ELSEIF (axis_source(idim).EQ.pgc_impose_axis) THEN
* ... imposed axes are handled later by GCF_IMPOSE_AXES
	      GOTO 500
	   ELSE
* ... it is evidently, a mergible axis ... continue on
	      CONTINUE
	   ENDIF	      
	ENDIF

* ignore axes with unknown limits
* 2/27/96 - kludge bug fix
* Dynamic axes and the logic inadequacies of FLESH_OUT_AXIS have
* created a problem. This check, which was here previously,
*	   IF ( cx_lo_ww(idim,com_cx) .EQ. unspecified_val8 ) GOTO 500
* was supposed to check if an axis has anything relevant to contribute. That
* logic was based on the fact that during grid determination subscript limits
* of 1:1,1:1,1:1,1:1 are supplied. Then FLESH_OUT_AXIS fills in the WW limits.
* But as the 7/92 "kludge" in FLESH_OUT_AXIS explains the logic is not reliable
* during grid determination. (A correct fix in that routine is to have a
* global Boolean indicating if Ferret is in its grid determination phase.)
* As a result, FLESH_OUT_AXIS may set the limits to unspecified ... counting
* on them to be filled in by the variable definition, itself. (Correct
* behavior for context determination, but not for grid determination).
* These unspecified limits were causing the axis to be ignored.
* On 2/27 we change this logic to use uvar_given to determine if this
* axis should be ignored.

* Made a special exception for mrg_ax being mnormal. In that case, even
* the unspecified limits are relevant. This is a porrly understood kludge
* fix to accomodate the fact that the grid is now initially set at all mnormal.
	   IF ( cx_lo_ww(idim,com_cx) .EQ. unspecified_val8
     .    .AND. mrg_ax .NE. mnormal				) THEN
	      IF ( cx_category(com_cx) .NE. cat_user_var ) GOTO 500
	      lims_gvn = uvar_given(idim, cx_variable(com_cx))
	      IF ( lims_gvn.GT. uvlim_gvn_xact
     .        .OR. lims_gvn.LT. uvlim_gvn_gnl ) GOTO 500
	   ENDIF

* 5/96 *sh*: have the limits of the new component been supplied?
* Previously the code simply used cx_given(idim,com_cx) to decide this
* but in doing so it missed the possibility that limits were given deeper
* inside of the component definition.
* Now we use expl_com_lim instead.
* 1/97: avoid uvlim_irrelevant mistakenly implying expl_com_lims is TRUE
	   expl_com_lim = cx_given(idim,com_cx)
	   IF ( nest ) expl_com_lim = expl_com_lim .OR. 
     .		    (     uvar_given(idim,com_uvar).LE.uvlim_gvn_xact     
     .		    .AND. uvar_given(idim,com_uvar).GE.uvlim_gvn_gnl  )
* (NOTE: might a similar test be needed on mrg_cx?)

* new axis same as merged one ?
	   same = com_ax .EQ. mrg_ax
! 5/96	   IF ( cx_given(idim,com_cx) .AND. cx_given(idim,mrg_cx) )
	   IF ( expl_com_lim .AND. cx_given(idim,mrg_cx) )
     .	      same  = same
     .		.AND. cx_lo_ss(com_cx,idim) .EQ. cx_lo_ss(mrg_cx,idim)
     .		.AND. cx_hi_ss(com_cx,idim) .EQ. cx_hi_ss(mrg_cx,idim)

* determine pointers into logic table
! 5/96	   IF ( cx_given(idim,com_cx) ) THEN
	   IF ( expl_com_lim ) THEN
	      IF ( same ) THEN
	        icom = 1
	      ELSE
	        icom = 3
	      ENDIF
	   ELSE
	      IF ( same ) THEN
	        icom = 5
	      ELSE
	        icom = 6
	      ENDIF
	   ENDIF
	   IF ( uvar_given(idim,uvar) .EQ. uvlim_irrelevant ) THEN
	      imrg = 5
	   ELSEIF ( mrg_ax .EQ. mnormal ) THEN
	      imrg = 4
	   ELSEIF ( cx_given(idim,mrg_cx) ) THEN
	      IF ( mrg_len .EQ. 1 ) THEN
	        imrg = 1
	      ELSE
	        imrg = 2
	      ENDIF
	   ELSE
	      imrg = 3
	   ENDIF
	   IF ( com_len.GT.1 .AND. icom.LT.5 ) icom = icom + 1

* update the merged cx_given value (added 1/97)
	cx_given(idim,mrg_cx) = cx_given(idim,mrg_cx)
     .			  .AND. expl_com_lim

********
* entire kludge block commented out 1/97. (Probably was needed previously
* because ax_choice(5,4) used to result in error -- changed to "NA" (no-op).
!* "irrelevant" axis patch (8/30/89): (mod 4/90 using MERGIFY_GRID_AXIS)
!* irrelevant component won't arrive here because it has unspecified_val8
!* if, however, the first item in the expression had limits flagged irrelevant
!* then the merged grid may still be flagged as irrelevant
!	   IF ( uvar_given(idim,uvar) .EQ. uvlim_irrelevant ) THEN
!* *sh* 10/95 arrive here for X axis of "x[x=0.1:10:.1] + y[y=101:110:1]"
!*  ... dont really understand why ... but need to "USE" com_ax or it gets
!*      deallocated.  Not sure if mrg_ax could ever dynamic but in case
!*      it seems safer to IF dynamic THEN DEALLo
! 	      IF ( com_ax .GT. max_lines ) CALL TM_USE_LINE(com_ax)
!	      IF ( mrg_ax .GT. max_lines ) CALL TM_DEALLO_DYN_LINE(mrg_ax)
!	      grid_line( idim, mrg_grd ) = com_ax
!	      cx_lo_ss( mrg_cx, idim ) = cx_lo_ss( com_cx, idim )
!	      cx_hi_ss( mrg_cx, idim ) = cx_hi_ss( com_cx, idim )
!	      CALL MERGIFY_GRID_AXIS( com_cx, idim, uvar )
!	      WRITE (6,*) 'MERGIFY_GRID_AXIS called from MERGE_GRID',
!     .				idim,icom,imrg
!	      GOTO 500
!	   ENDIF	      
********

* take appropriate merging action
*		  EXP  IMP WRN1   ***   NA  WRN2
	   GOTO ( 100, 200,  90, 9000, 400, 190 ) ax_choice(icom,imrg)

* give a warning before imposing ambiguous explicit limits
* (use grid name as a flag to prevent multiple warnings)
 90	   IF ( grid_name(mrg_grd) .EQ. 'UVAR' ) CALL WARN
     .			('Ambiguous coordinates on '//
     .			ww_dim_name(idim)//' axis: '//
     .			uvar_text(uvar)(:TM_LENSTR1(uvar_text(uvar))))
	   grid_name(mrg_grd) = 'AMBG'

* impose explicit limits from component that gave them (e.g. U[K=5])
 100	   IF ( com_ax .GT. max_lines ) CALL TM_USE_LINE(com_ax)
	   IF ( mrg_ax .GT. max_lines ) CALL TM_DEALLO_DYN_LINE(mrg_ax)
	   grid_line( idim, mrg_grd ) = com_ax
	   cx_lo_ss( mrg_cx, idim ) = cx_lo_ss( com_cx, idim )
	   cx_hi_ss( mrg_cx, idim ) = cx_hi_ss( com_cx, idim )
	   GOTO 400

* give a warning before imposing ambiguous implicit limits
!(Should really use the WARN subroutine ...)
 190	   IF ( grid_name(mrg_grd) .EQ. 'UVAR' ) CALL WARN
     .			('Ambiguous coordinates on '//
     .			ww_dim_name(idim)//' axis: '//
     .			uvar_text(uvar)(:TM_LENSTR1(uvar_text(uvar))))
	   grid_name(mrg_grd) = 'AMBG'

* impose an axis with implicit coordinates (inferred from the context)
 200	   IF ( com_ax .GT. max_lines ) CALL TM_USE_LINE(com_ax)
	   IF ( mrg_ax .GT. max_lines ) CALL TM_DEALLO_DYN_LINE(mrg_ax)
	   grid_line( idim, mrg_grd ) = com_ax

* save implications about how limits are given in this component
* if the component is a user-defined variable then consider the implications
* of its definition
* com_class encodes the way limits are given in this component
* uv_class is an intermediate for computing uvar_lim
* ... first the a/b option in the joint_limit_match table
 400	   IF ( cx_lo_ww(idim,com_cx) .EQ. unspecified_val8 ) GOTO 500  ! 1/97
	   IF ( uvar_given(idim,uvar) .EQ. uvlim_need_xact ) GOTO 500
* get limits-region class of new component
	   IF ( nest ) THEN
	      com_class = NEST_REGION_CLASS( idim, com_cx, com_uvar )
	   ELSE
	      com_class = REGION_CLASS( idim, com_cx )
	   ENDIF	      

	   IF ( uvar_given(idim,uvar) .EQ. uvlim_irrelevant ) THEN ! 1/97
* nothing meaningful has been merged in so far
	      uv_class = com_class				   ! 1/97
	   ELSE
* get limits-region class of so-far-merged components
	      mrg_class = MOD( uvar_given(idim,uvar), 4 )
* how does the new component effect the merged class ?
	      uv_class = joint_match( mrg_class, com_class )
	   ENDIF

* if the new component or any previous lacks given limits so does the result
* e.g. LIST/K=1:5 U - U[K=3]
* (the /K limits are lacking in the overall expression)
	   IF ( uvar_given(idim,uvar).GE. uvlim_needed
     .	   .OR. .NOT.cx_given(idim,com_cx)
     .	   .OR. .NOT.INHERITED_LIMITS(idim,com_cx) ) THEN
	      uvar_given(idim,uvar) = uv_class + uvlim_needed
	   ELSE
	      uvar_given(idim,uvar) = uv_class
	   ENDIF
	
 500	CONTINUE

* deallocate any implicit (temporary) grid that was created for this context
	IF ( cx_has_impl_grid(com_cx) ) THEN
	   IF ( mode_diagnostic )  CALL DIAG_OP
     .		('dealloc ', isact_class_impl_grid, com_grd, 0)
	   cx_has_impl_grid(com_cx) = .FALSE.
	   CALL TM_DEALLO_DYN_GRID( com_grd )
	ENDIF

* successfully merged
	status = ferr_ok
	RETURN

* error exit
 5000	RETURN
 9000	CALL ERRMSG( ferr_internal, status, 'mrg_grd_err', *5000 )
	END
