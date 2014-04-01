	SUBROUTINE GET_AUX_VAR_CONTEXT
     .		      ( var_cx,
     .			aux_var_cx, 
     .			aux_arg_num,
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
* V685 - new based upon GET_VAR_CONTEXT

* this routine gets the context of the auxiliary variable arguments (e.g."depth")
* within parenthese in the format   VAR[gz(depth)=zax@ave]
*  

* it is performed in two steps: first get the context of the main variable
* VAR.  Then use the context of the parent variable as the default for the
* auxiliary variable ("depth")

	IMPLICIT NONE

	include	'ferret.parm'
	include	'errmsg.parm'
	include	'interp_stack.parm'
        include 'tmap_dims.parm'
	include	'xcontext.cmn'
	include	'xvariables.cmn'
	include 'xprog_state.cmn'
	include 'xtext_info.cmn'
        include 'xdset_info.cmn_text'
	external xdset_info_data
        include 'xtm_grid.cmn_text'
        external xgt_grid_data

* calling argument declarations:
	INTEGER	 var_cx, aux_var_cx, aux_arg_num, status

* internal variable declarations:
	INTEGER	 TM_LENSTR1, AUX_AXIS_FROM_ARG,
     .		 the_axis, var_axis, aux_var_axis, i, j, idim
	CHARACTER*128 VAR_CODE, var_name, aux_var_name

* initialize
	status = ferr_ok

* copy the var_context to form the base for the aux_var context
	CALL TRANSFER_CONTEXT(var_cx, aux_var_cx)	

* remove regridding requests - aux vars must always be on native grids
	cx_has_impl_grid( aux_var_cx ) = .FALSE.
	cx_unstand_grid ( aux_var_cx ) = .FALSE.
	cx_grid		( aux_var_cx ) = unspecified_int4  ! cannot assume parent grid 

* which auxiliary var is the subject of interest?
* e.g. for "temp[gz(depth)=zax]" the argument "depth" is found in the z dim
	the_axis = AUX_AXIS_FROM_ARG(var_cx, aux_arg_num)

* pull the auxiliary variable identity from the parent context
	cx_category(aux_var_cx) = cx_aux_cat(the_axis, var_cx)
	cx_variable(aux_var_cx) = cx_aux_var(the_axis, var_cx)

* remove traces of further aux var dependency
	cx_naux         ( aux_var_cx ) = 0
	DO idim = 1, nferdims
	   cx_aux_cat (idim,aux_var_cx) = unspecified_int4  ! just bookkeeping
	   cx_aux_var (idim,aux_var_cx) = unspecified_int4  ! just bookkeeping
	   cx_aux_stat(idim,aux_var_cx) = paux_stat_na
	ENDDO

* insert bad data flag for variable
	CALL GET_CONTEXT_BAD_FLAG( aux_var_cx )

* insert data type for variable
	CALL KNOWN_DATA_TYPE( aux_var_cx )

* determine the defining grid of the auxiliary variable
	CALL GET_CONTEXT_GRID ( aux_var_cx, status )
	IF ( status .NE. ferr_ok ) GOTO 5000
! check <0 to trap "hack" from GET_CONTEXT_MODS
	IF ( cx_grid(aux_var_cx) .EQ. unspecified_int4
     .	.OR. cx_grid(    var_cx) .EQ. unspecified_int4
     .  .OR. cx_grid(    var_cx) .LE. 0                  ) RETURN 
! return here is only reasonable during getgrid phase.  Not during eval phase.

* set up the axes of the aux variable
* only the regridding axis(s) of the aux grid may differ from the parent
	DO 500 idim = 1, nferdims
	   aux_var_axis = grid_line(idim,cx_grid(aux_var_cx))
	   var_axis     = grid_line(idim,cx_grid(    var_cx))
	   IF (idim .EQ. the_axis) THEN
	      IF (aux_var_axis .EQ. mnormal) GOTO 5300 
* ... set up to get the full index range on the regrid axes of the aux var
              cx_lo_ss ( aux_var_cx, idim )  = unspecified_int4
              cx_hi_ss ( aux_var_cx, idim )  = unspecified_int4
              cx_lo_ww ( idim, aux_var_cx )  = unspecified_val8
              cx_hi_ww ( idim, aux_var_cx )  = unspecified_val8
              cx_by_ss ( idim, aux_var_cx )  = .FALSE.       !for bookkeeping
              cx_given( idim, aux_var_cx )   = .FALSE.
	   ELSEIF (aux_var_axis .EQ. var_axis) THEN
	      GOTO 500		! already fleshed out through TRANSFER_CONTEXT
	   ELSEIF (aux_var_axis .NE. mnormal) THEN
	      GOTO 5400
	   ENDIF
	   CALL FLESH_OUT_AXIS( idim, aux_var_cx, status )  ! always ferr_ok
	   IF (status .NE. ferr_ok) 
     .	      CALL ERRMSG( ferr_internal, status, 'aux_var_ax', *5000)
 500	CONTINUE

	RETURN

* error exit
 5000	RETURN
 5300	aux_var_name = VAR_CODE(cx_category(aux_var_cx),
     .			        cx_variable(aux_var_cx) )
	var_name     = VAR_CODE(cx_category(var_cx),
     .			        cx_variable(var_cx) )
	i = TM_LENSTR1(aux_var_name)
	j = TM_LENSTR1(    var_name)
	CALL ERRMSG( ferr_invalid_command, status,
     .          'auxiliary variable '//
     .		aux_var_name(:i)//' lacks the '//
     .		ww_dim_name(the_axis)//' axis '//
     .	        'it needs to regrid '//var_name(:j), *5000 )
 5400	aux_var_name = VAR_CODE(cx_category(aux_var_cx),
     .			        cx_variable(aux_var_cx) )
	var_name     = VAR_CODE(cx_category(var_cx),
     .			        cx_variable(var_cx) )
	i = TM_LENSTR1(aux_var_name)
	j = TM_LENSTR1(    var_name)
	CALL ERRMSG( ferr_invalid_command, status,
     .          'auxiliary variable '//
     .		aux_var_name(:i)//' has a '//
     .		ww_dim_name(idim)//' axis '//
     .	        'not found on variable '//var_name(:j), *5000 )
	END
