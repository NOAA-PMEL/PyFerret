	SUBROUTINE ERASE_DSET ( dset )

* delete all memory variables with data from a particular data set
* eliminate all context references to it, too

* programmer - steve hankin
* NOAA/PMEL, Seattle, WA - Tropical Modeling and Analysis Program
* written for VAX computer under VMS operating system
*
* revision 0.0 - 4/3/86
* V200:   6/9/89 - deleted variables flagged in mr_protected
*	10/25/89 - moved memory-resident purging to PURGE_DSET
* V411:  7/95 - Delete variables created with  LET/D=this_dset

#ifdef unix
	include	'ferret_cmn/ferret.parm'
	include 'ferret_cmn/xvariables.cmn'
	include 'ferret_cmn/xcontext.cmn'
#else
	INCLUDE	'FERRET_CMN:FERRET.PARM'
	INCLUDE 'FERRET_CMN:XVARIABLES.CMN'
	INCLUDE 'FERRET_CMN:XCONTEXT.CMN'
#endif

* calling argument declarations:
	INTEGER	dset

* local variable declarations:
	INTEGER mr, ii

* eliminate memory resident data
	CALL PURGE_DSET( dset )

* delete LET-defined variables that belong to this data set
	DO 100 ii = 1, max_uvar
	   IF (uvar_dset(ii) .EQ. dset) uvar_num_items(ii) =
     .					uvar_deleted
 100	CONTINUE


* eliminate context information depending on this data set
	DO 200 ii = 1, max_context
	   IF ( cx_data_set( ii ) .EQ. dset ) THEN
	      cx_data_set( ii ) = unspecified_int4
	      cx_grid( ii ) = unspecified_int4
	   ENDIF
 200	CONTINUE

	RETURN
	END

