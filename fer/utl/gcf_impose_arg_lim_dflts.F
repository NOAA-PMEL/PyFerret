	SUBROUTINE GCF_IMPOSE_ARG_LIM_DFLTS( vname, arg,
     .					     cx_in, cx_out, uvar, status )

* This routine is used in setting the context used for **obtaining**
* the component arguments to a grid changing function.
* This is the so-called "upstream" passing of context.

* The input context, cx_in, is the limits requested for the result.
* For example, in
*	LET a = GCFCN( b )
*	LIST/I=1:5 a[J=5:10]
* the input context would contain I=1:5 and J=5:10.
* The decision of what defaults to set when requesting the argument ("b")
* is the business of this routine.

* V450: 1/97

#ifdef unix
	include 'tmap_format/tmap_dims.parm'
	include	'tmap_format/xtm_grid.cmn_text'
	external xgt_grid_data
	include 'ferret_cmn/ferret.parm'
	include 'ferret_cmn/errmsg.parm'
	include 'ferret_cmn/grid_chg_fcns.parm'
	include	'ferret_cmn/xcontext.cmn'
	include	'ferret_cmn/xvariables.cmn'
	include 'ferret_cmn/xfr_grid.cmn'       ! w/ mgrid_abstract
	include 'ferret_cmn/xtext_info.cmn'
#else
	INCLUDE 'TMAP_FORMAT:TMAP_DIMS.PARM'
	INCLUDE	'TMAP_FORMAT:XTM_GRID.CMN'
	INCLUDE 'FERRET_CMN:FERRET.PARM'
	INCLUDE 'FERRET_CMN:ERRMSG.PARM'
	INCLUDE	'FERRET_CMN:XCONTEXT.CMN'
	INCLUDE	'FERRET_CMN:XVARIABLES.CMN'
	INCLUDE 'FERRET_CMN:XFR_GRID.CMN'       ! w/ mgrid_abstract
	INCLUDE 'FERRET_CMN:XTEXT_INFO.CMN'
#endif

* calling argument declarations
	INTEGER		arg, cx_in, cx_out, uvar, status
	CHARACTER*(*)	vname

* local variable declarations
	LOGICAL CGRID_AXIS
	INTEGER	TM_LENSTR1, ISUBSCRIPT, KNOWN_GRID,
     .		gcfcn, idim, arg_grd, mods_cx, tmp_var_cx,
     .		lo_off(4), hi_off(4), action(4), uvgrid,
     .          WHICH_GCFCN

* initialize
	CALL TRANSFER_CONTEXT ( cx_in, cx_out )

* When this routine is called the grid of the argument is not yet known
* We must know the grid in order to apply offsets
* These two stack levels are just needed temporarily to learn the grid
	CALL STACK_PTR_UP ( cx_stack_ptr, max_context, status )
	IF ( status .NE. ferr_ok ) GOTO 5000
	tmp_var_cx = cx_stack_ptr
	CALL TRANSFER_CONTEXT ( cx_in, tmp_var_cx )
	CALL STACK_PTR_UP ( cx_stack_ptr, max_context, status )
	IF ( status .NE. ferr_ok ) GOTO 5000
	mods_cx = cx_stack_ptr
	CALL INIT_CONTEXT_MODS ( mods_cx )

* get variable and attached modifications
	CALL PARSE_VAR_NAME ( vname, tmp_var_cx, mods_cx, status )
	IF ( status .NE. ferr_ok ) GOTO 5000
	
* determine the defining grid of the argument
	CALL GET_CONTEXT_GRID ( tmp_var_cx, status )
	IF ( status .NE. ferr_ok ) GOTO 5000
	arg_grd = cx_grid(tmp_var_cx)
	cx_grid(cx_out) = arg_grd

* can't do more if the grid isn't known
	cx_stack_ptr = cx_stack_ptr - 2
	IF ( arg_grd .EQ. unspecified_int4 ) RETURN	! ??? error ???

* which grid changing function does this uvar have?
	gcfcn = WHICH_GCFCN(uvar)

* get the grid of the result of this function
	uvgrid = KNOWN_GRID(cx_data_set(cx_in), cat_user_var, uvar)

* What did the function creator specify about this argument?
	CALL GCF_GET_ARG_LIMS( gcfcn, arg, cx_out, uvgrid,
     .				lo_off, hi_off, action, status)
	IF ( status .NE. ferr_ok ) RETURN

* loop through the axes
	DO 500 idim = 1,4

	   IF ( action(idim) .EQ. pset_to_unspec ) THEN
* set the axis limits to unspecified
	      cx_lo_ss(cx_out,idim)      = unspecified_int4
	      cx_hi_ss(cx_out,idim)      = unspecified_int4
	      cx_by_ss(idim,cx_out)      = .TRUE.
	      cx_given(idim,cx_out)      = .FALSE.	! needed ?
!	      cx_trans (idim,cx_out)     = trans_no_transform
!	      cx_trans_arg(idim,cx_out)  = bad_val4

	   ELSEIF (lo_off(idim).NE.0 .OR. hi_off(idim).NE.0) THEN
* pass along the axis limits
	      IF ( cx_by_ss(cx_out,idim) ) THEN
* ... expand the subscript range by the offsets requested
	         IF ( cx_hi_ss(cx_out,idim).NE.unspecified_int4 ) THEN
	            cx_lo_ss(cx_out,idim) = cx_lo_ss(cx_out,idim)
     .					  + lo_off(idim)
	            cx_hi_ss(cx_out,idim) = cx_hi_ss(cx_out,idim)
     .					  + hi_off(idim)
		 ENDIF
	      ELSE
* ... convert the world coordinate range to subscript limits. Then
* ... expand the subscript range by the offsets requested
	         IF ( cx_hi_ww(idim,cx_out).NE.unspecified_val8 ) THEN
	            cx_lo_ss(cx_out,idim) =
     .				ISUBSCRIPT(cx_lo_ww(idim,cx_out),
     .					   arg_grd, idim, round_up)
	            cx_hi_ss(cx_out,idim) = 
     .				ISUBSCRIPT(cx_hi_ww(idim,cx_out),
     .					   arg_grd, idim, round_dn)
	            cx_lo_ss(cx_out,idim) = cx_lo_ss(cx_out,idim)
     .					  + lo_off(idim)
	            cx_hi_ss(cx_out,idim) = cx_hi_ss(cx_out,idim)
     .					  + hi_off(idim)
		 ENDIF
	         cx_by_ss(idim,cx_out) = .TRUE.
	      ENDIF
	   ENDIF

 500	CONTINUE

* successfully merged
	status = ferr_ok
	RETURN

* error exit
 5000	RETURN
	END

