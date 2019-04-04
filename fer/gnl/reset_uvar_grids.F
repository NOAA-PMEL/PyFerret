	SUBROUTINE RESET_UVAR_GRIDS

* Clear stored user grid info before or after a
* grave-accent evaluation.

* 3/2019  Ansley Manke

        IMPLICIT NONE
        include 'tmap_dims.parm'
	include 'ferret.parm'
	include 'xvariables.cmn'
	include 'xcontext.cmn'
	include 'xtm_grid.cmn_text'


	INTEGER i, uvar, dset, idim
	REAL	GET_LINE_COORD, rval

	IF (.NOT. cx_only) RETURN

	DO i = 1, uvar_cxo_len

	   rval = GET_LINE_COORD ( linemem(uvar_cx_only_lm)%ptr, i )
	   uvar = rval
	   rval = GET_LINE_COORD ( lineedg(uvar_cx_only_lm)%ptr, i )
	   dset = rval

	   IF (uvar .LE. 0) CYCLE
	   
	   DO idim = 1, nferdims
	      uvar_given ( idim, uvar ) = uvlim_irrelevant

	      uvar_lo_ss ( uvar, idim )  = unspecified_int4   !for bookkeeping
	      uvar_hi_ss ( uvar, idim )  = unspecified_int4
	      uvar_lo_ww ( idim, uvar )  = unspecified_val8
	      uvar_hi_ww ( idim, uvar )  = unspecified_val8
	      uvar_by_ss ( idim, uvar )  = .FALSE.          

	   ENDDO

* remove all of the grids associated with this uvar. 
* DEALLO_UVAR_GRIDS does the free operation

	   CALL DEALLO_UVAR_GRIDS(uvar) 
c	   CALL NCF_FREE_UVAR_GRID_LIST( dset, uvar)


	ENDDO

	RETURN
	END
