	SUBROUTINE GET_VAR_CONTEXT
     .		      ( full_var_name, 
     .			default_cx,
     .			var_cx,
     .			status )

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
* given a full variable specification of the form:
*			(eg.) TEMP[X=170E:170W,K=+1D,T=188979]
* return in var_cx context information describing the exact variable instance
* including variable, data set, grid, region and plane

* programmer - steve hankin
* NOAA/PMEL, Seattle, WA - Tropical Modeling and Analysis Program
* written for VAX computer under VMS operating system
*
* revision 0.0 - 10/15/86
* revision 0.1 - 3/23/87 - added new TMAP library: time axes, file var. names,
*						   bad data flag
*			 - context mods from var name applied IN PARSE_VAR_NAME
* revision 0.2 - 5/5/88  - FINISH_CONTEXT_MODS --> FLESH_OUT_CONTEXT
*			   added CONFINE_CONTEXT
* V200:  6/22/89 - eliminated GT_CONTEXT_PLANE
*		 - added oversizing of contexts
*		 - added test for irrelevant data set
* V500: 10/20/98 *jc* - added ADD_RDC_XFORM
* V510: 8/99 *sh* - g=unknown_uvar is allowed now. If happening, then skip
*	final steps of context getting
* V533: 6/01 *sh* - set data type
* V603: 5/07 *acm*  Fix for bug 1475, before confine_axis check cx_by_ss(idim,cx)
*                   If .true. check for transformation reducing range to a pt
*       *acm* 3/12 Add E and F dimensions (use nferdims in tmap_dims.parm)
* V685 4/13 *sh* parse the auxiliary regridding context terms ("*aux*"), too
*	         and don't report errors in dependency_tree_mode
* V690 12/13 *sh* bug fix when variable name unrecognized in VTREE mode
* V693 10/14 *acm* fix ticket 2206: If the grid number is a placeholder, don't 
*                 go call flesh_out_axis etc (See the comment in get_context_mods, where
*                 the grid is an encoding of the start/end limits of the regrid spec)
* V695+ 5/15 *sh* report unrecognized aux var unless in dependency_tree_mode

	IMPLICIT NONE

	include	'ferret.parm'
	include	'errmsg.parm'
        include 'tmap_dims.parm'
	include 'xfr_grid.cmn'
	include	'xcontext.cmn'
	include	'xvariables.cmn'
	include	'xprog_state.cmn'
        include 'xdset_info.cmn_text'
        include 'xdependency_tree.cmn'
	include 'xtm_grid.cmn_text' 
        external xdset_info_data
#        include "tmap_dset.parm"

* calling argument declarations:
	INTEGER		default_cx, var_cx, status
	CHARACTER*(*)	full_var_name

* internal variable declarations:
	LOGICAL		unreported_error
	INTEGER		mods_cx, idim, i, istart, iend, cat, var, iarg,
     .			ignore

* initialize
	unreported_error = .FALSE.
	CALL TRANSFER_CONTEXT ( default_cx, var_cx )

* find a context slot for the modifications that may be attached to the var name
	CALL STACK_PTR_UP ( cx_stack_ptr, max_context, status )
	IF ( status .NE. ferr_ok ) GOTO 5000
	mods_cx = cx_stack_ptr
	CALL INIT_CONTEXT_MODS ( mods_cx )

* get variable and attached modifications
	IF ( dependency_tree_mode ) allowed_err = .TRUE.  ! SHOW VAR/TREE
	CALL PARSE_VAR_NAME ( full_var_name, var_cx, mods_cx, status )
	allowed_err = .FALSE.
	IF ( status .NE. ferr_ok ) THEN
* ... when merely determining a dependency tree unknow var is OK
	   IF ( dependency_tree_mode ) THEN
	      cx_variable(var_cx) = unspecified_int4
	      cx_category(var_cx) = unspecified_int4
	      unreported_error = .TRUE.
	   ELSE
	      GOTO 5000
	   ENDIF
	ENDIF

* in dependency tree mode bail out of implicit grid created by GET_CONTEXT_MODS
	IF ( dependency_tree_mode ) THEN
	   IF (cx_has_impl_grid(var_cx)) THEN
	      IF (cx_impl_grid(var_cx).NE.grd_stk_ptr )
     .				CALL WARN('crptn_get_var_ctxt')
	      grid_name(grd_stk_ptr) = char_init16
	      grd_stk_ptr = grd_stk_ptr + 1
	      cx_has_impl_grid(var_cx) = .FALSE.
	   ENDIF
	ENDIF

* if explicit auxiliary regridding variables were detected, identify them
	IF (cx_naux(var_cx).GT.0 .AND. .NOT.cx_implicit_aux(var_cx)) THEN
	   iarg = 0
* ... cx_naux always matches the number of valid cx_aux_arg_start values
	   DO idim = 1, nferdims
	      IF (cx_aux_arg_start(idim, var_cx) .NE. unspecified_int4) THEN
	         iarg = iarg + 1
	         cx_valid_aux(iarg,var_cx) = .TRUE.
* ... make sure there are no square brackets
* ... sq brtk contents like D= would require hard dependencies on the aux vars
	         DO i = cx_aux_arg_start(idim,var_cx),
     .		        cx_aux_arg_end  (idim,var_cx) 
	            IF (full_var_name(i:i) .EQ. '[') GOTO 5100
                 ENDDO
* ... identify the aux variable name
* ... btw note: cx_aux_arg_start is EQUIV to cx_aux_var. Ditto cx_aux_cat.
	         istart = cx_aux_arg_start(idim,var_cx)
	         iend   = cx_aux_arg_end  (idim,var_cx)
	         CALL FIND_VAR_NAME( cx_data_set(var_cx),
     .				     full_var_name(istart:iend),
     .				     cat,
     .				     var )
	         IF (var .NE. munknown_var_name) THEN
	            cx_aux_cat (idim,var_cx) = cat
	            cx_aux_var (idim,var_cx) = var
	            cx_aux_stat(idim,var_cx) = paux_stat_used
	         ELSEIF ( dependency_tree_mode ) THEN
	            cx_valid_aux(iarg,var_cx) = .FALSE. ! a name not found
	            IF (.NOT. unreported_error) THEN
	               status = ferr_unknown_auxvar
		       unreported_error = .TRUE.
	            ENDIF
	         ELSE
	            GOTO 5200
	         ENDIF	
	      ENDIF
	   ENDDO
	ENDIF
	IF (unreported_error) THEN
	   CALL STACK_PTR_DN(cx_stack_ptr,cx_stack_ptr_base,ignore) ! mods_cx
	   GOTO 5000
	ENDIF

* below here is work that is needed for evaluations;  not for dependency tree
	IF ( dependency_tree_mode ) GOTO 100  ! 12/31

* 8/99 If G=unknown_user_var was given then exit now.
* It is recognizable because the text position is encoded in the grid number.
* This should happen ONLY when this is called from IS_UVAR_GRID
	IF ( cx_unstand_grid(var_cx)
     . .AND. cx_grid(var_cx) .LT. unspecified_int4 ) GOTO 100

* apply the hidden @RDC transform if appropriate
	CALL ADD_RDC_XFORM( var_cx, status)
	IF ( status .NE. ferr_ok ) GOTO 5000

* insert bad data flag for variable
	CALL GET_CONTEXT_BAD_FLAG( var_cx )

* insert data type for variable
	CALL KNOWN_DATA_TYPE( var_cx )

* determine the defining grid
	CALL GET_CONTEXT_GRID ( var_cx, status )
	IF ( status .NE. ferr_ok ) GOTO 5000

* can't do more if the grid isn't known
	IF ( cx_grid(var_cx) .LE. unspecified_int4 ) GOTO 100

* data set info may be irrelevant
* indicate that now so memory searches will be correct
	IF ( cx_category(var_cx) .EQ. cat_user_var ) THEN
	   IF ( .NOT.uvar_need_dset(cx_variable(var_cx)) ) THEN
	      cx_data_set(var_cx) = pdset_irrelevant
*  ... flag that all data set info is contained within the variable
	      cx_dset_gvn(var_cx) = .TRUE.
	   ENDIF
	ENDIF

c Fix for bug 1475
c FLESH_OUT_AXIS has a check for whether there is a transformation that
c reduces the range to a point and if so sets cx_by_ss(idim,cx)=.FALSE.

** NOTE: This change causes bug 1748. The index range for the transform
** doesn't get checked in confine_context.

* flesh out the context on each axis with grid-related positions
* seeing if cx_by_ss(idim,cx) should really be in effect
	DO 11 idim = 1, nferdims
	   IF ( cx_by_ss(idim,var_cx) ) THEN
              CALL FLESH_OUT_AXIS( idim, var_cx, status )
	      IF ( status .NE. ferr_ok ) GOTO 5000
           ENDIF
 11	CONTINUE

* confine limits to grid
	CALL CONFINE_CONTEXT ( var_cx, status )
	IF ( status .NE. ferr_ok ) GOTO 5000

* flesh out the context on each axis with grid-related positions
	DO 10 idim = 1, nferdims
	   CALL FLESH_OUT_AXIS( idim, var_cx, status )
	   IF ( status .NE. ferr_ok ) GOTO 5000
 10	CONTINUE

* enlarge the context ? (contour plots need data extending beyond plot region)
	IF ( cx_ovsiz( var_cx) .NE. ovsiz_off ) CALL OVSIZ_CX( var_cx )

* relinquish the stack space used for the modifications
 100	CALL STACK_PTR_DN ( cx_stack_ptr, cx_stack_ptr_base, status )
	IF ( status .NE. ferr_ok ) GOTO 5000
	RETURN

* error exit
 5000	RETURN

 5100	CALL ERRMSG( ferr_syntax, status,
     .          'square bracket modifiers may not be used with '//
     .          'auxiliary regridding variables:'//pCR
     .		//full_var_name, *5000 )

 5200	CALL ERRMSG( ferr_unknown_auxvar, status,
     .          'regridding variable "'//
     .		full_var_name(istart:iend)//
     .          '" in '//full_var_name, *5000 )

	END
