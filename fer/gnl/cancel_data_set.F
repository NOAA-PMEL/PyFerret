	SUBROUTINE CANCEL_DATA_SET( iset, status )

* close and delete all dependening on a data set

* programmer - steve hankin
* NOAA/PMEL, Seattle, WA - Tropical Modeling and Analysis Program
* written for VAX computer under VMS operating system
*
* V200:	10/20/89 - extracted from XEQ_CANCEL with enhancements to accept /ALL
*		   and data set names
*	 12/8/89 - if saved_dset is cancelled then indicate that
* Unix/RISC port 8/91 *sh*
*     bug fix: calling var, iset, was used as an internal loop index
* Linux Port 1/97 *kob* - Added a preprocessor include for tmap_dset.parm
*			  because it needed a preprocessor.

#ifdef unix
	include 'tmap_format/tmap_dims.parm'
#	include "tmap_format/tmap_dset.parm"
	include 'tmap_format/tmap_errors.parm'
	include	'ferret_cmn/ferret.parm'
	include 'ferret_cmn/errmsg.parm'
	include 'ferret_cmn/xdset_parms.cmn'
	include 'ferret_cmn/xprog_state.cmn'
#else
	INCLUDE 'TMAP_FORMAT:TMAP_DIMS.PARM'
	INCLUDE 'TMAP_FORMAT:TMAP_DSET.PARM'
	INCLUDE 'TMAP_FORMAT:TMAP_ERRORS.PARM'
	INCLUDE	'FERRET_CMN:FERRET.PARM'
	INCLUDE 'FERRET_CMN:ERRMSG.PARM'
	INCLUDE 'FERRET_CMN:XDSET_PARMS.CMN'
	INCLUDE 'FERRET_CMN:XPROG_STATE.CMN'
#endif

* calling argument declarations:
	INTEGER	iset, status

* local variable declarations:
	INTEGER	iclim, iclim_user

* close and cancel the data set
	CALL TM_CLOSE_SET( iset, status )
	IF ( status .NE. merr_ok ) RETURN

* no longer the saved data set ?
	IF ( iset .EQ. saved_dset ) saved_dset = unspecified_int4

* delete any memory variables from this data set
* ( necessary because their pointers will lead to incorrect data set 
*   information and possibly misleading output )
	CALL ERASE_DSET( iset )

* delete all user defined data from memory for similar reasons
* (note: this is unnecessarily strict - could be refined in the future)
	CALL PURGE_ALL_UVARS

* was this a climatological air temp field ?
	DO 100 iclim = 1, max_airt_clim
 	   IF ( dp_airt_dset(iclim) .EQ. iset ) THEN
	      dp_airt_dset(iclim) = unspecified_int4
	      GOTO 200
	   ENDIF
 100	CONTINUE
* ... no - we're done
	RETURN

* ... yes - delete any variables that may have used it
*     ( a precaution against mixed up clim. data sets )
 200	DO 300 iclim_user = 1, maxdsets
	   IF ( dp_clim_air_sst(iclim_user) .EQ. iclim )
     .                                   CALL ERASE_DSET( iset )
 300	CONTINUE

	RETURN
	END
