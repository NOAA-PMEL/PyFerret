	SUBROUTINE PURGE_ALL_UVARS

* eliminate all stored user variables from memory

* programmer - steve hankin
* NOAA/PMEL, Seattle, WA - Tropical Modeling and Analysis Program

* V200:  6/29/89
* Unix/RISC port - 11/91 - bug fix: re-initialization of uvars wasn't complete
*                          for redetermination of grid

* V420 - use new routine DEALLO_UVAR_GRIDS to both deallocate dynamic
*	 grids associated with a uvar and clean out any memory-cached 
*	 results that may depend on this grid

#ifdef unix
	include 'ferret_cmn/ferret.parm'
	include 'ferret_cmn/errmsg.parm'
	include 'ferret_cmn/xvariables.cmn'
#else
	INCLUDE 'FERRET_CMN:FERRET.PARM'
	INCLUDE 'FERRET_CMN:ERRMSG.PARM'
	INCLUDE 'FERRET_CMN:XVARIABLES.CMN'
#endif

* internal variable declarations:
	INTEGER	uvar, i

	DO 500 uvar = 1, max_uvar                      
	   IF ( uvar_num_items( uvar ) .EQ. uvar_deleted ) GOTO 500  

* eliminate all memory resident instances of this variable
	   CALL PURGE_USER_VAR( uvar )

* back to initial defaults (see INIT_UVAR)
* (eliminate all grid information that may have been rendered incorrect)
	   CALL DEALLO_UVAR_GRIDS(uvar)

*       bug fix: 11/91 *************
* must reset ALL of the state variables as set up in INIT_UVAR
* by default assume the definition requires an external data set specification
        uvar_need_dset( uvar ) = .TRUE.

* by default assume that all limits are irrelevant
        DO 320 i = 1, 4
 320           uvar_given(i,uvar) = uvlim_irrelevant
* end of bug fix: 11/91 *************


 500	CONTINUE

	RETURN
	END
