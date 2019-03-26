	SUBROUTINE RESET_UVAR_GRIDS (isp_start)

* Clear stored user grid info before or after a
* grave-accent evaluation.

* 3/2019  Ansley Manke

        IMPLICIT NONE
        include 'tmap_dims.parm'
	include 'ferret.parm'
	include 'xvariables.cmn'

	INTEGER isp_start

	INTEGER i, uvar, idim

	DO i = isp_start, max_intrp
	   uvar = is_uvar(i)
	   IF (uvar.GT.0 .AND. uvar.LT.max_uvar) THEN
	      DO idim = 1, nferdims
	        uvar_given ( idim, uvar ) = uvlim_irrelevant
	        uvar_lo_ss ( uvar, idim ) = unspecified_int4
	        uvar_hi_ss ( uvar, idim ) = unspecified_int4
	      ENDDO

	      CALL DEALLO_UVAR_GRIDS(uvar) 

	   ENDIF
	ENDDO

	RETURN
	END
