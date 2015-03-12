	SUBROUTINE CANCEL_DATA_SET( iset, is_all, status )

*  This software was developed by the Thermal Modeling and Analysis
*  Project(TMAP) of the National Oceanographic and Atmospheric
*  Administration's (NOAA) Pacific Marine Environmental Lab(PMEL),
*  hereafter referred to as NOAA/PMEL/TMAP.
*
*  Access and use of this software shall impose the following
*  obligations and understandings on the user. The user is granted the
*  right, without any fee or cost, to use, copy, modify, alter, enhance
*  and distribute this software, and any derivative works thereof, and
*  its supporting documentation for any purpose whatsoever, provided
*  that this entire notice appears in all copies of the software,
*  derivative works and supporting documentation.  Further, the user
*  agrees to credit NOAA/PMEL/TMAP in any publications that result from
*  the use of this software or in any product that includes this
*  software. The names TMAP, NOAA and/or PMEL, however, may not be used
*  in any advertising or publicity to endorse or promote any products
*  or commercial entity unless specific written permission is obtained
*  from NOAA/PMEL/TMAP. The user also understands that NOAA/PMEL/TMAP
*  is not obligated to provide the user with any support, consulting,
*  training or assistance of any kind with regard to the use, operation
*  and performance of this software nor to provide the user with any
*  updates, revisions, new versions or "bug fixes".
*
*  THIS SOFTWARE IS PROVIDED BY NOAA/PMEL/TMAP "AS IS" AND ANY EXPRESS
*  OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED
*  WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
*  ARE DISCLAIMED. IN NO EVENT SHALL NOAA/PMEL/TMAP BE LIABLE FOR ANY SPECIAL,
*  INDIRECT OR CONSEQUENTIAL DAMAGES OR ANY DAMAGES WHATSOEVER
*  RESULTING FROM LOSS OF USE, DATA OR PROFITS, WHETHER IN AN ACTION OF
*  CONTRACT, NEGLIGENCE OR OTHER TORTUOUS ACTION, ARISING OUT OF OR IN
*  CONNECTION WITH THE ACCESS, USE OR PERFORMANCE OF THIS SOFTWARE. 
*
*
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
* V533: *sh* 6/01 - close set after deletions to avoid "%%" diagnostics
* V685 *acm* 4/13 - If its a member of an aggregation, also close the aggregation.
*                   Pass in whether this was "all", if so dont issue the
*                   note for canceling a member dataset

	include 'tmap_dims.parm'
#	include "tmap_dset.parm"
	include 'tmap_errors.parm'
	include	'ferret.parm'
	include 'errmsg.parm'
	include 'xdset_parms.cmn'
	include 'xprog_state.cmn'
	include 'xdset_info.cmn_text'
	external xdset_info_data
	include 'xrisc.cmn'
	include	'xtm_grid.cmn_text'
	external xgt_grid_data

* calling argument declarations:
	LOGICAL is_all
	INTEGER	iset, status

* local variable declarations:
	LOGICAL hidden
	INTEGER TM_LENSTR1
	INTEGER	iclim, iclim_user, i, nmemb, imemb, membset, 
     .          mlen, slen1, slen2, cset, close_it, nv, igrid, 
     .          ens_line, agg_dim, iens
	CHARACTER*20 agg_msg
	CHARACTER*8 TM_FMT, set1, set2


* Recursive loop. If canceling a dset that is a member of an ensemble 
* or ensembles, we will come back here and cancel the ensemble set too.

	iens = 1
	cset = iset

100	CONTINUE

	IF (cset .EQ. 0) GOTO 300

* Will want to cancel the ensemble axis
	ens_line = 0
	IF (ds_type(cset) .EQ. 'ENS') THEN
	   agg_dim = 5
	   DO 10 nv = 1, maxvars
 	      IF (ds_var_setnum(nv) .EQ. cset) THEN
	          igrid = ds_grid_number(nv)
		  ens_line = grid_line(agg_dim, igrid)
		  GOTO 15  ! only 1 ensemble axis per ens. dataset
	      ENDIF
   10	   CONTINUE
   15	   CONTINUE
	ENDIF

* Will want to cancel the Forecast axis

	IF (ds_type(cset) .EQ. 'FCT') THEN
	   agg_dim = 6
	   DO 11 nv = 1, maxvars
 	      IF (ds_var_setnum(nv) .EQ. cset) THEN
	          igrid = ds_grid_number(nv)
		  ens_line = grid_line(agg_dim, igrid)
		  GOTO 16  ! only 1 ensemble axis per ens. dataset
	      ENDIF
   11	   CONTINUE
   16	   CONTINUE
   
	ENDIF
* delete any memory variables from this data set
* ( necessary because their pointers will lead to incorrect data set 
*   information and possibly misleading output )
	CALL ERASE_DSET( cset )

* delete all user defined data from memory for similar reasons
* (note: this is unnecessarily strict - could be refined in the future)
	CALL PURGE_ALL_UVARS
	
* If canceling an aggregation then reset the hidden setting to the 
* default for member datasets.  Check on ens_line finds cases where
* an ensemeble datset was created but has no variables.

	IF ((ds_type(cset).EQ.'ENS' .OR. ds_type(cset).EQ.'FCT')
     .      .AND. ens_line.GT.0) THEN

* Get number of aggregation members
	   CALL CD_GET_AGG_DSET_INFO (cset, nmemb, status)

* Look through the aggregation members.

	   hidden = .FALSE.
	   DO 20 imemb = 1, nmemb
	      CALL CD_GET_AGG_DSET_MEMBER (cset, imemb, membset, status)
	      IF (status .NE. ferr_ok) GOTO 20
	      IF (ds_hide(membset)) hidden = .TRUE.
	      ds_hide(membset) = .FALSE.
  20	   CONTINUE
  
	   IF (hidden) THEN
	     risc_buff = 
     .	   'Cancel aggregate dataset. Un-hiding hidden member datasets'
	     CALL WARN(risc_buff)
	   ENDIF

	ENDIF

* close and cancel the data set
	CALL TM_CLOSE_SET( cset, status )
	IF ( status .NE. merr_ok ) RETURN

* Cancel the ensemble axis
	IF ( ens_line .NE. 0) CALL DELETE_ONE_LINE (ens_line, status)

* Was the original dataset to be closed a member of an aggregation?  If so cancel 
* the aggregation set. If we implement other aggregations, set agg_msg depending 
* on the ds_type direction.

	cset = iset
	close_it = 0
	DO 50 i = iens, maxdsets
	   IF ((ds_type(i).EQ.'ENS' .OR. ds_type(i).EQ.'FCT')) THEN
	      agg_msg = 'ensemble'
	      IF (ds_type(cset).EQ.'FCT') agg_msg = 'forecast aggregation'
	      mlen = TM_LENSTR1(agg_msg)

* Get number of aggregation members
	      CALL CD_GET_AGG_DSET_INFO (i, nmemb, status)

* Look through the aggregation members.

	      DO 40 imemb = 1, nmemb
	         CALL CD_GET_AGG_DSET_MEMBER (i, imemb, membset, status)
		 IF (status .NE. ferr_ok) GOTO 40

* Was cset a member of this aggregation? If so close the aggregation

		 IF (imemb .EQ. cset) THEN 
		    set1 = TM_FMT (float(cset), 8, 8, slen1)
		    set2 = TM_FMT (float(i), 8, 8, slen2)

		    IF (.NOT. is_all) THEN
		       WRITE (risc_buff, 1000) set1(:slen1), agg_msg(:mlen), 
     .                     agg_msg(:mlen),set2(:slen2)
		       CALL WARN(risc_buff)
		    ENDIF
		    
* ! Found a match. Go close this aggregation set
		    close_it = i
		    cset = i
		    iens = i+1
		    GOTO 60 

		 ENDIF
  40	      CONTINUE
	   ENDIF
  50	CONTINUE
  60	CONTINUE
  
	IF (close_it .GT. 0) GOTO 100

 1000	FORMAT ('Dataset ', A,' was a member of an ', A,'. Canceling ',
     .        A,', dataset ', A)

  300	CONTINUE

* no longer the saved data set ?
	IF ( iset .EQ. saved_dset ) saved_dset = unspecified_int4

* was this a climatological air temp field ?
	DO 400 iclim = 1, max_airt_clim
 	   IF ( dp_airt_dset(iclim) .EQ. iset ) THEN
	      dp_airt_dset(iclim) = unspecified_int4
	      GOTO 500
	   ENDIF
 400	CONTINUE
* ... no - we're done
	RETURN

* ... yes - delete any variables that may have used it
*     ( a precaution against mixed up clim. data sets )
 500	DO 600 iclim_user = 1, maxdsets
	   IF ( dp_clim_air_sst(iclim_user) .EQ. iclim )
     .                                   CALL ERASE_DSET( iset )
 600	CONTINUE

        RETURN
	END
