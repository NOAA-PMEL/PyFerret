	SUBROUTINE UNPROTECT_CMND_DATA

* unprotect (flag as not in use) the data from the most recent command

* programmer - steve hankin
* NOAA/PMEL, Seattle, WA - Tropical Modeling and Analysis Program

* V230: 7/27/92 - removed code from GET_CMND_DATA

#ifdef unix
	include	'ferret_cmn/ferret.parm'
	include 'ferret_cmn/xvariables.cmn'
#else
	INCLUDE	'FERRET_CMN:FERRET.PARM'
	INCLUDE 'FERRET_CMN:XVARIABLES.CMN'
#endif

* calling argument declarations
* ...

* local variable declarations
	INTEGER		nu, mr

* remove protections from the data grids
	DO 400 nu = 1, num_uvars_in_cmnd
	   mr = is_mr( nu )
	   IF ( mr_protected(mr) .NE. mr_deleted ) 
     .                                CALL MR_NOT_IN_USE( mr )
 400	CONTINUE

! temp diagnostic - last mem var fetched should never be in-use by this stage
 	IF ( mr_protected( mr ) .NE. mr_perm_protected
     . .AND. mr_protected( mr ) .NE. mr_not_protected
     . .AND. mr_protected( mr ) .NE. mr_temporary
     . .AND. mr_protected( mr ) .NE. mr_deleted )
     .			WRITE (6,*) 'get_cmnd_prot_err'

 	RETURN
	END
