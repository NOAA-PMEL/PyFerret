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
* V490: 11/97 - bug fix: when wiping out context axis wipe ww lims, too
* V491:  3/98 - bug fix: crashed because pseudo-variables did not have
*		valid grids when this routine was called. Changed to use
*		**result** grid to convert world coords to subscripts when
*		offsets need to be applied -- argument grid not needed.
*		Hugely simplified this routine!
*	Note: the "offset" code in this routine is largely untested!

#ifdef unix
	include 'tmap_format/tmap_dims.parm'
	include	'tmap_format/xtm_grid.cmn_text'
	external xgt_grid_data
	include 'ferret_cmn/ferret.parm'
	include 'ferret_cmn/errmsg.parm'
	include 'ferret_cmn/grid_chg_fcns.parm'
	include	'ferret_cmn/xcontext.cmn'
#else
	INCLUDE 'TMAP_FORMAT:TMAP_DIMS.PARM'
	INCLUDE	'TMAP_FORMAT:XTM_GRID.CMN'
	INCLUDE 'FERRET_CMN:FERRET.PARM'
	INCLUDE 'FERRET_CMN:ERRMSG.PARM'
	INCLUDE	'FERRET_CMN:XCONTEXT.CMN'
#endif

* calling argument declarations
	INTEGER		arg, cx_in, cx_out, uvar, status
	CHARACTER*(*)	vname		! unused after 3/98

* local variable declarations
	INTEGER	TM_LENSTR1, ISUBSCRIPT, KNOWN_GRID,
     .		gcfcn, idim, arg_grd, mods_cx, tmp_var_cx,
     .		lo_off(4), hi_off(4), action(4), uvgrid,
     .          WHICH_GCFCN

* initialize
	CALL TRANSFER_CONTEXT ( cx_in, cx_out )

* which grid changing function does this uvar have?
	gcfcn = WHICH_GCFCN(uvar)

* get the grid of the result of this function (user variable)
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
	      cx_lo_ww(idim,cx_out)      = unspecified_val8	! 11/97
	      cx_hi_ww(idim,cx_out)      = unspecified_val8	! 11/97
!	      cx_trans (idim,cx_out)     = trans_no_transform
!	      cx_trans_arg(idim,cx_out)  = bad_val4

	   ELSEIF (lo_off(idim).NE.0 .OR. hi_off(idim).NE.0) THEN
* pass along the axis limits
	      IF ( cx_by_ss(idim,cx_out) ) THEN
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
     .					   uvgrid, idim, round_up)
	            cx_hi_ss(cx_out,idim) = 
     .				ISUBSCRIPT(cx_hi_ww(idim,cx_out),
     .					   uvgrid, idim, round_dn)
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
! 5000	RETURN
	END

