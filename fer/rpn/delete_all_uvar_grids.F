	SUBROUTINE DELETE_ALL_UVAR_GRIDS

* delete all the grids that have been determined for user variables

* programmer - steve hankin
* NOAA/PMEL, Seattle, WA - Tropical Modeling and Analysis Program
*
* V230:  8/10/92 - needed for DEFINE REGION which may change the implied
*                  context info of uvars using let x = var[@reg] notation

#ifdef unix
	include 'ferret_cmn/ferret.parm'	
	include 'ferret_cmn/xvariables.cmn'
#else
	INCLUDE 'FERRET_CMN:FERRET.PARM'	
	INCLUDE 'FERRET_CMN:XVARIABLES.CMN'
#endif

* calling argument declarations:
* ...

* internal variable declarations:
	INTEGER	i, uvar

* loop through the user-defined variables
	DO 500 uvar = 1, max_uvar
	   IF ( uvar_num_items( uvar ) .EQ. uvar_deleted ) GOTO 500

* return to default that all limits are irrelevant
	   DO 100 i = 1, 4
 100	   uvar_given(i,uvar) = uvlim_irrelevant

* reset grid to unknown for all data sets and clear dynamic grid info
	   CALL DEALLO_UVAR_GRIDS(uvar)

 500	CONTINUE

 	RETURN
	END
