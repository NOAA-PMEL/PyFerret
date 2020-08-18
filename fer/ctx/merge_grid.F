	SUBROUTINE MERGE_GRID(	iarg, its_gc, com_text,
     .				mrg_cx, com_cx, uvar, status )

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
* V491:   2/23/98 - 1/97 check at line 400 for unspec cx_lo_ww introduced a bug
*                 in cases like LET p=i[i=1:5]; LET q=i[i=1:5]; LET pq=p*q;
*                 uvar_given for pq was set to uvlim_irrelevant instead of 
*                 uvlim_gvn_gnl
*         2/28/98 - further repair to resolve problem that re-appeared in
*                 err490_l_ave.jnl: use "expl_com_lim" (explicit component
*                 limits) **consistently** instead of cx_given to know whether
*                 a new component does or does not possess its own limits
* V500: *sh* 6/99 -- BUG COMMENT ONLY!! NOT FIXED. See err500_expl_lims.jnl
*		  and find "6/99" in code below
* V510: *sh* 12/99 - support for uvlim_hidden (EF caching)
*	*sh* 12/99 - set the uvar_lo/hi limits along with uvar_given
*	*sh* 12/99 - bug fix: rules for merging embodied in joint_match were
*			wrong when new component is xact
*	*sh*  4/00 - all grids and axes now maintain use counts 
* V520  *sh* 6/00 - complete the support for uvlim_hidden. Was treated just
*		    like uvlim_irrelevant in determining uvar_given
* V533  *sh* 7/01 - determine data type of result
* V64+ *acm 12/09 - Fix bug 1710: In the grid axis selection table, ax_choice(4,1)
*                   changed from Not Possible to Not Applicable
*      *acm* 6D Ferret: use nferdims rather than 4 for dimension indexing
* v685 *acm* 4/13   Call WARN for warning message
* v685 *acm* 5/13   Fix ticket 2051, setting uvar_given(idim,uvar) when the
*                   user-variable has multiple sub-expressions
* v690 *sh* 2/14  Determine the curvilinear (aux var) dependencies of the merged
*		  expression
* V694+*acm*5/15  Fix for ticket 2273: return= shape, *size for expressions built 
*                 of other exprns.
* V695 *acm* 6/16 Fix ticket 1602. Do not substitute abstract axes for normal axes
*                 when merging a constant.
* V720: 4/17 *sh* special behaviors defining DSG grids
* V72  *acm* 6/17 For the above change, ticket 1602, and also for comment #3 of 2313, 
*                 do the check differently. If the the grid of the component is 
*                 all-abstract (and the variable is a constant), do not add abstract 
*                 axes to the merged grid.
* v745 *acm* 3/19 Changes to the fix for ticket 2051 (issue 1323). It causes bug 1916.
* V76  1/20 *acm* working with Point-type dsg data
* V76  2/20 *acm* Regridding Point-type dsg data, restore value of com_grd before deallocating

        IMPLICIT NONE
	include 'tmap_dims.parm'
	include 'implicit.parm'    ! has pline_not_dsg
#include "tmap_dset.parm"
	include 'grid_chg_fcns.parm'
	include	'xtm_grid.cmn_text'
	external xgt_grid_data
	include 'ferret.parm'
	include 'errmsg.parm'
	include	'interp_stack.parm'
	include	'xcontext.cmn'
	include	'xvariables.cmn'
	include	'xprog_state.cmn'
	include	'xtext_info.cmn'
	include 'xdset_info.cmn_text'
	include	'xfr_grid.cmn'

* calling argument declarations
	LOGICAL		its_gc
	INTEGER		iarg, mrg_cx, com_cx, uvar, status
	CHARACTER	com_text*(*)

* local variable declarations
	LOGICAL		INHERITED_LIMITS, ITSA_PURE_SUM_VAR, TM_ITSA_DSG,
     .			ITSA_COMPATIBLE_DSG_GRID, ITSA_DSG_POINTS,
     .			same, nest, expl_com_lim, abstgrid, normabst,
     .			var_is_const, its_dsg, its_cmpnd
	INTEGER		CX_DIM_LEN, TM_LENSTR1, REGION_CLASS,
     .			NEST_REGION_CLASS,  WHICH_GCFCN, GCF_RTN_TYPE,
     .			TM_DSG_DSET_FROM_GRID,
     .			mrg_grd, mrg_ax, mrg_len, com_grd, com_ax, com_len,
     .			idim, i, icom, imrg, com_uvar, com_class, mrg_class,
     .			uv_class, ax_choice(6,5), joint_match(0:3,0:3),
     .			lims_gvn, gcfcn, axis_source(nferdims), uvar_test,
     .			naux, blen, mabstract, iline, dsg_dset, dsg_orient, 
     .			obsdimlen, fline
	CHARACTER	VAR_CODE*128, buff*128

* local parameter declarations
	INTEGER		EXP,   IMP,   WRN1,   NP,   NA,  WRN2
	PARAMETER     ( EXP=1, IMP=2, WRN1=3, NP=4, NA=5, WRN2=6 ) 
	INTEGER		gnl,   lo,   lohi,   xact
	PARAMETER     ( gnl=0, lo=1, lohi=2, xact=3 )

* FERRET DOCUMENTATION: merging grids for an expression
*
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
*                   expl_com_lim       .NOT.expl_com_lim 
*                same          diff      same   diff 
*  size:       1     >1      1     >1
*	    ------------------------------------------
*1       1  | EXP  |  *** | EXP | EXP |  IMP  | IMP  |
*    given  ------------------------------------------
*2      >1  | ***  |  NA  | NA  | WRN1|  IMP  | IMP  |
*           ------------------------------------------
*3.NOT.given| NA   |  NA  | NA  | NA  |  NA   | WRN2 |
*           ------------------------------------------
*4  normal  | NA** |  *** | EXP | EXP |  NA*  | IMP  | (NA* was *** pre 1/97, NA** was *** pre 12/09)
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
*	uvlim_hidden	- limits hidden inside on another axis: NEW=FFT(v)
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
*   P |	xact	|      lohi	lohi    lohi	  xact	 |
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
     .		(ax_choice(i,4),i=1,6) / NA ,NP ,EXP,EXP, NA ,IMP  /,
     .		(ax_choice(i,5),i=1,6) / EXP,EXP,EXP,EXP, IMP,IMP  /
! pre 1/97 .	(ax_choice(i,4),i=1,6) / NP , NP, EXP, EXP, NP , IMP  /
! pre 12/09.	(ax_choice(i,4),i=1,6) / NP , NP ,EXP, EXP, NA ,IMP  /,

	DATA	(joint_match(i,0),i=0,3) /  gnl,   lo, lohi, gnl  /,
     .		(joint_match(i,1),i=0,3) /   lo,   lo, lohi, lo   /,
     .		(joint_match(i,2),i=0,3) / lohi, lohi, lohi, lohi /,
     .		(joint_match(i,3),i=0,3) / lohi, lohi, lohi, xact /

**************************************************************************
* is this a grid changing variable? (1/97)
	IF (its_gc) THEN
	   gcfcn = WHICH_GCFCN(uvar)
	   CALL GCF_GET_AXIS_SRCS(gcfcn, iarg, axis_source)
	ENDIF

* initialize
	mrg_grd = cx_grid( mrg_cx )
	com_grd = cx_grid( com_cx )

* is this a DSG variable?
! DSGTBD: DSG grids are handled as small hacks in this complex routine
! If this leads to trouble consider a separate dsg_merge_grid subroutine
! DSG merges are very simple - E is always present.  The "dsg_orient"
! axis may or may not be.  There is just one acceptable axis for either
! of these slots.  All other axes are always normal.
! A DSG context violates gridded norms in that will carry ww coodinate
! limits on mnormal axes
!
! When merging DSG grids that have info in the obs direction, then use the dsg_xlate_grid
! storing that in mgrid_buff, a scratch grid used just locally in this routine.
! dsg_xlate_grid(dsg_dset) used to translate to world coords, but only for X,Y,Z,T

	its_dsg = TM_ITSA_DSG(com_grd) !!.AND. .NOT.ITSA_DSG_POINTS(com_cx)
	IF (its_dsg) THEN
	   dsg_dset = TM_DSG_DSET_FROM_GRID(com_grd)
	   IF (ITSA_DSG_POINTS(com_cx)) com_grd = dsg_xlate_grid(dsg_dset)
	   IF (.NOT.ITSA_COMPATIBLE_DSG_GRID(mrg_grd, com_grd)) GOTO 5100

* Grid has XYZT directions - not just an instance grid?
	   normabst = .TRUE.
	   DO idim = x_dim, t_dim
	      IF (grid_line(idim, com_grd) .NE. mnormal) normabst = .FALSE.
	   ENDDO

* If the grid has axes in the XYZT directions, then they represent coordiates
* and we want to use dsg_xlate_grid for those dimensions.

	   IF (.NOT.normabst .AND. dsg_xlate_grid(dsg_dset) .GT. mgrid_buff) THEN 
	      
	      CALL TM_DSG_FACTS( com_grd, dsg_orient, obsdimlen, fline, its_dsg, its_cmpnd )
	      DO idim = x_dim, t_dim
	         grid_line(idim, mgrid_buff)  = grid_line(idim, dsg_xlate_grid(dsg_dset))
	      ENDDO
	      DO idim = t_dim+1, nferdims
	         grid_line(idim, mgrid_buff)  = grid_line(idim, com_grd)
	      ENDDO

	      com_grd = mgrid_buff  ! dummy grid used just locally
	   ENDIF
	ELSE
	   dsg_dset = cx_data_set(com_cx)
	   IF (ITSA_DSG_POINTS(com_cx)) com_grd = dsg_xlate_grid(dsg_dset)
	ENDIF
	   
* do we have nested user-defined variables ?
	nest = cx_category( com_cx ) .EQ. cat_user_var
	IF ( nest ) THEN
	   com_uvar = cx_variable( com_cx )
	   var_is_const = uvar_item_type( uvar_num_items(com_uvar), com_uvar )
     .          .EQ. alg_constant
	ENDIF

* Is the com_grd all  abstract ?  If so we won't add abstract axes willy-nilly
* to the grid
	mabstract = grid_line( 1, mgrid_abstract )
	normabst = .TRUE.
	DO idim = 1, nferdims
	   iline = grid_line(idim,com_grd)
	   IF (iline.NE.mabstract) normabst = .FALSE.
	ENDDO

* for merged context to be data set independent all components must be so
	IF ( cx_category(com_cx) .EQ. cat_pseudo_var ) THEN
* ... pseudo-vars are merged only if explicit "G=".  Was it G=var or G=grid ?
	   cx_dset_gvn(mrg_cx) = cx_dset_gvn(mrg_cx)
     .				.AND. .NOT.cx_grid_needs_dset
	ELSE
	   cx_dset_gvn(mrg_cx) = cx_dset_gvn(mrg_cx) .AND.
     .	(cx_dset_gvn(com_cx).OR.cx_data_set(com_cx).EQ.pdset_irrelevant)
	ENDIF

* data type of result
	IF (its_gc) THEN
* ... can get result data type only when types of all arguments are known
*     cuz it may be inherited from an argument
	  IF (iarg .EQ. 1) THEN
	    i = GCF_RTN_TYPE(gcfcn, com_cx)
	    IF (i .EQ. pfcn_rtn_string) THEN
	      cx_type(mrg_cx) = ptype_string
	    ELSE
	      cx_type(mrg_cx) = ptype_float
	    ENDIF
	  ENDIF
	ELSE
	  IF (uvar_num_items(uvar) .EQ. 1) THEN 
	    cx_type(mrg_cx) = cx_type(com_cx)   ! v="a" or v=5

* ... special HACK to allow string addition: "a" + "b"
	  ELSEIF (ITSA_PURE_SUM_VAR(uvar)) THEN
	    IF (cx_type(mrg_cx) .EQ. unspecified_int4) THEN
	        cx_type(mrg_cx) = cx_type(com_cx)  ! inherit first arg type
	    ELSEIF (cx_type(mrg_cx) .NE. cx_type(com_cx)) THEN
	        cx_type(mrg_cx) = ptype_float      ! unmatched arg types
	    ENDIF

	  ELSE
	    cx_type(mrg_cx) = ptype_float       ! expression float unless GC
	  ENDIF
	ENDIF

* loop through the axes

* 2/2014 - pass the aux var dependencies from the component to the merged result
* merged result will contain a conflict wherever components point to different
* aux variables for the same axis
	naux = 0
	DO idim = 1, nferdims
	   IF (cx_aux_var(idim,com_cx) .NE. unspecified_int4) THEN
	      naux = naux + 1
	      IF (cx_aux_stat(idim,mrg_cx).EQ.paux_stat_conflict ) THEN
* ... if previous merges had a conflict, it doesn't go away
	         CONTINUE
	      ELSEIF (cx_aux_stat(idim,mrg_cx).EQ.paux_stat_na) THEN
	         cx_aux_var (idim,mrg_cx) = cx_aux_var (idim,com_cx)
	         cx_aux_cat (idim,mrg_cx) = cx_aux_cat (idim,com_cx)
	         cx_aux_stat(idim,mrg_cx) = cx_aux_stat(idim,com_cx)
	      ELSEIF ( ( cx_aux_var(idim,mrg_cx)
     .	            .NE. cx_aux_var(idim,com_cx) ) 
     .		  .OR. ( cx_aux_cat(idim,mrg_cx)
     .		    .NE. cx_aux_cat(idim,com_cx) )  ) THEN
	         cx_aux_stat(idim,mrg_cx) = paux_stat_conflict
	         cx_aux_var (idim,mrg_cx) = unspecified_int4
	         cx_aux_cat (idim,mrg_cx) = unspecified_int4
	         IF (mode_diagnostic) THEN
* ... "Z axis auxiliary coordinates of expression in conflict.
*     "Variable TEMP coordinates are from DEPTH1"
*     "Other components indicate DEPTH2 
	             buff = ww_dim_name(idim)//
     .	' axis auxiliary coordinates of expression are in conflict.'
	             CALL SPLIT_LIST(pttmode_explct, ttout_lun, buff, 0)
	             buff = 'Variable '//VAR_CODE(cx_category(com_cx),
     .						  cx_variable(com_cx))
	             blen = TM_LENSTR1(buff)
	             buff(blen+1:) = 'coordinates are from '//
     .			VAR_CODE(cx_aux_cat(idim,com_cx),
     .				 cx_aux_var(idim,com_cx) )//'.'
	             CALL SPLIT_LIST(pttmode_explct, ttout_lun, buff, 0)
	             buff = 'Other components indicate '//
     .			VAR_CODE(cx_aux_cat(idim,mrg_cx),
     .				 cx_aux_var(idim,mrg_cx) )//'.'
	             CALL SPLIT_LIST(pttmode_explct, ttout_lun, buff, 0)
	          ENDIF
	       ELSEIF (cx_aux_stat(idim,mrg_cx).EQ.paux_stat_needed) THEN
* ... new compoonent uses same aux as previously merged
	         cx_aux_stat(idim,mrg_cx) = paux_stat_needed
	       ENDIF
	   ELSEIF (cx_aux_stat(idim,mrg_cx) .NE. paux_stat_na
     .	     .AND. cx_aux_stat(idim,mrg_cx) .NE. paux_stat_passed ) THEN
	       naux = naux + 1
	   ENDIF
	   IF (cx_aux_stat(idim,com_cx) .EQ. paux_stat_needed)
     .			cx_aux_stat(idim,mrg_cx) = paux_stat_needed
	ENDDO
	cx_naux(mrg_cx) = naux

* pass the axes and theirimplied dependencies to the merged result
	DO 500 idim = 1, nferdims

* get axes and lengths of data along them
	   com_ax  = grid_line ( idim, com_grd )
	   mrg_ax  = grid_line ( idim, mrg_grd )
	   com_len = CX_DIM_LEN( idim, com_cx  )
	   mrg_len = CX_DIM_LEN( idim, mrg_cx  )

* 5/17 ignore normal axis of new component  (but why only for DSG?)
	   IF (its_dsg .AND. com_ax.EQ.mnormal) GOTO 500

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
* the unspecified limits are relevant. This is a poorly understood kludge
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
*       by performing a more careful check on uvar_given than was done
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
	   IF ( uvar_given(idim,uvar) .EQ. uvlim_irrelevant
     .	   .OR.  uvar_given(idim,uvar) .EQ. uvlim_hidden  ) THEN
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
! 6/99 NEXT LINE IS NOT AN ADEQUATE TEST!! com_len may in relity be GT 1, but
! buried within a uvar def'n. uvar_given tells that limits are buried, but
! NOT whether the limits are GT 1.
!  example: "v" in  SST[x=180] + v, where v=X[GX=xax,I=1:10]
	   IF ( com_len.GT.1 .AND. icom.LT.5 ) icom = icom + 1

* update the merged cx_given value (added 1/97)
	cx_given(idim,mrg_cx) = cx_given(idim,mrg_cx)
     .			  .AND. expl_com_lim

* Do not substitute an abstract axis for normal axes in the grid 
* when merging a constant

	IF (com_ax.EQ.mabstract .AND. mrg_ax.EQ.mnormal) THEN
c	   IF (.NOT. (normabst) .AND. var_is_const) GOTO 400
	   IF (normabst .AND. var_is_const) GOTO 400
	ENDIF

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
 100	   CALL TM_USE_LINE(com_ax)
	   CALL TM_DEALLO_DYN_LINE(mrg_ax)
	   grid_line( idim, mrg_grd ) = com_ax
	   cx_lo_ss( mrg_cx, idim ) = cx_lo_ss( com_cx, idim )
	   cx_hi_ss( mrg_cx, idim ) = cx_hi_ss( com_cx, idim )
	   GOTO 400

* give a warning before imposing ambiguous implicit limits

 190	   IF ( grid_name(mrg_grd) .EQ. 'UVAR'
     .		.OR. grid_name(mrg_grd) .EQ. 'AMBG' ) CALL WARN
     .			('Ambiguous coordinates on '//
     .			ww_dim_name(idim)//' axis: '//
     .			uvar_text(uvar)(:TM_LENSTR1(uvar_text(uvar))))
	   grid_name(mrg_grd) = 'AMBG'

* impose an axis with implicit coordinates (inferred from the context)
 200	   CALL TM_USE_LINE(com_ax)
	   CALL TM_DEALLO_DYN_LINE(mrg_ax)
	   grid_line( idim, mrg_grd ) = com_ax

* save implications about how limits are given in thi uvar_given(idim,uvar) .EQ. uvlim_s component
* if the component is a user-defined variable then consider the implications
* of its definition
* com_class encodes the way limits are given in this component
* uv_class is an intermediate for computing uvar_lim
* ... first the a/b option in the joint_limit_match table

* ... first screen out irrelevant cases
* 2/23/98 - for a component which is a user var, uvar_given tells if it
*           is irrelevant  
*        For a component which is not, cx_lo_ww was already used prior to 2/98
 400	   IF ( nest ) THEN
	      IF (uvar_given(idim,com_uvar).EQ.uvlim_irrelevant) THEN
	         GOTO 500
* (6/0) ... pass hidden limits if no specification preceded it
	      ELSEIF (uvar_given(idim,com_uvar).EQ.uvlim_hidden) THEN
	         IF ( uvar_given(idim,uvar).EQ.uvlim_irrelevant )
     .				uvar_given(idim,uvar) = uvlim_hidden
	         GOTO 500
	      ENDIF
	   ELSE
	      IF (cx_lo_ww(idim,com_cx) .EQ. unspecified_val8) GOTO 500 ! 1/97
	   ENDIF

* If a previous component gave us uvlim_need_xact, we just kept that setting. 
* for the entire expression. This was the cause of ticket 2051. Use 
* uvlim_need_lohi instead for the tests below.

*  For 2273, we do not want to change the value of uvar_test here. The 
*  examples in ticket 2051 still give correct resu;ts, but the RETURN= examples 
*  in the report for 2273 did not.  Perhaps changes for the `RETURN=isready` 
*  have allowed cause  this to work better?

	   uvar_test =  uvar_given(idim,uvar)
	   IF ( .NOT.cx_only .AND. uvar_test .EQ. uvlim_need_xact ) GOTO 500

* get limits-region class of new component
	   IF ( nest ) THEN
	      com_class = NEST_REGION_CLASS( idim, com_cx, com_uvar )
	   ELSE
	      com_class = REGION_CLASS( idim, com_cx )
	   ENDIF

	   IF ( uvar_given(idim,uvar) .EQ. uvlim_irrelevant 
     .	   .OR. uvar_given(idim,uvar) .EQ. uvlim_hidden) THEN ! 1/97
* nothing meaningful has been merged in so far
	      uv_class = com_class				   ! 1/97
	   ELSE
* get limits-region class of so-far-merged components
	      mrg_class = MOD( uvar_test, 4 )
* how does the new component effect the merged class ?
	      uv_class = joint_match( mrg_class, com_class )
	   ENDIF

* if the new component or any previous lacks given limits so does the result
* e.g. LIST/K=1:5 U - U[K=3]
* (the /K limits are lacking in the overall expression)
!    2/28/98: replaced:  	   .OR. .NOT.cx_given(idim,com_cx)
	   IF (its_dsg .AND. idim.NE.f_dim) THEN
! *sh* these settings are educated guesses -- concepts don't apply to DSGs
	      IF (idim.EQ.e_dim) THEN
	         uvar_given(idim,uvar) = uvlim_needed
	      ELSEIF (dsg_coord_var(idim,dsg_dset).EQ.int4_init) THEN
	         uvar_given(idim,uvar) = uvlim_irrelevant
	      ELSE
	         uvar_given(idim,uvar) = uvlim_needed
	      ENDIF
	   ELSEIF ( uvar_given(idim,uvar).GE. uvlim_needed
     .	   .OR. .NOT.expl_com_lim
     .	   .OR. .NOT.INHERITED_LIMITS(idim,com_cx) ) THEN
	      uvar_given(idim,uvar) = uv_class + uvlim_needed
	   ELSE
	      uvar_given(idim,uvar) = uv_class
	      IF ( cx_given(idim,com_cx) ) THEN
	         CALL TRANSFER_CX_AXIS_TO_UV( idim, com_cx, uvar)
	      ELSEIF ( expl_com_lim ) THEN
	         CALL TRANSFER_UV_AXIS_TO_UV( idim, com_uvar, uvar)
	      ENDIF
	   ENDIF
	
 500	CONTINUE

* deallocate any implicit (temporary) grid that was created for this context
	IF ( cx_has_impl_grid(com_cx) ) THEN
	   IF ( mode_diagnostic )  CALL DIAG_OP
     .		('dealloc ', isact_class_impl_grid, com_grd, 0)
	   cx_has_impl_grid(com_cx) = .FALSE.
	   com_grd = cx_grid( com_cx )  ! if DSG, may have changed this
	   CALL TM_DEALLO_DYN_GRID( com_grd )
	ENDIF

* successfully merged
	status = ferr_ok
	RETURN

* error exit
 5000	RETURN
 9000	CALL ERRMSG( ferr_internal, status, 'mrg_grd_err', *5000 )

 5100	CALL ERRMSG( ferr_grid_definition, status,
     .		    "feature data must be regridded"//
     .		    " to combine it with gridded data", *5000)


	END
