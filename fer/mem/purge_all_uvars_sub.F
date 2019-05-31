	SUBROUTINE PURGE_ALL_UVARS_SUB( uvar)
	
* eliminate all stored user variables from memory
* pulled out of purge_all_uvars

* 3/2019  Ansley Manke

        IMPLICIT NONE
        include 'tmap_dims.parm'
	include 'ferret.parm'
	include 'xvariables.cmn'

	INTEGER uvar, i
	
	CALL PURGE_USER_VAR( uvar )

* back to initial defaults (see INIT_UVAR)
* (eliminate all grid information that may have been rendered incorrect)
	CALL DEALLO_UVAR_GRIDS(uvar)

* must reset ALL of the state variables as set up in INIT_UVAR
* by default assume the definition requires an external data set specification
        uvar_need_dset( uvar ) = .TRUE.

* by default assume that all limits are irrelevant
        DO i = 1, nferdims
	   uvar_given(i,uvar) = uvlim_irrelevant
        ENDDO

	RETURN
	END
