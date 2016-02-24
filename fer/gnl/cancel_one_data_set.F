	SUBROUTINE CANCEL_ONE_DATA_SET( dset, status )

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
* close a single dataset
* if this dataset is an aggregation remove its special axes

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
* V685 *acm* 4/13 - If its a member of an aggregation, also close the agg
* V698 *sh* 1/16 - complexity removed to the parent routine, CANCEL_DATA_SET

* calling arguments
*  dset      -  (input) dataset number to be closed
*  status    - (output) return status

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
	INTEGER	dset, status

* local variable declarations:
	LOGICAL NC_GET_ATTRIB_STRING, got_it
	INTEGER TM_GET_LINENUM
	INTEGER	iclim, iclim_user, i, nmemb, imemb, membset, 
     .          ivar, axis, varid, parent,
     .          memb_cat, memb_dset, memb_grid, iline, memb_var,
     .          attid, attype, attlen, attoutflag
	CHARACTER axname*12, attname*12

* local parameter declarations
	LOGICAL do_warn
	INTEGER max_slen
	PARAMETER (do_warn = .TRUE.,
     .		   max_slen = 12)


* since ensembles create so much dataset proliferation, need a diagnostic
* message to see what is being canceled
	IF (mode_diagnostic) CALL SPLIT_LIST(pttmode_ops, ttout_lun,
     .		' canceling dset '//ds_des_name(dset), 0 )

* delete any memory variables from this data set
* ( necessary because their pointers will lead to incorrect data set 
*   information and possibly misleading output )
	CALL ERASE_DSET( dset )

* If canceling an E or F aggregation then ...
* reduce the use-count on uvar grids associated with LET/D defs in agg
* member datasets. These were bumped up to ensure they stayed undeleted.
	IF (ds_type(dset).EQ.'ENS' .OR. ds_type(dset).EQ.'FCT') THEN
	   CALL FREE_AGG_UVAR_GRIDS(dset, status)
	   IF (status .NE. ferr_ok) GOTO 5000
	ENDIF

* If canceling an F aggregation then ...
* de-allocate special axes TF_LAG_F and TF_CAL_T
* get the names of these special axes from the attribute pointers
	IF (ds_type(dset).EQ.'FCT') THEN
	   CALL CD_GET_VAR_ID (dset, '.', varid, status)
	   CALL CD_GET_VAR_ATT_ID (dset, varid, '_cal_t_ax', attid, status)
	   IF (status.EQ.atom_not_found) GOTO 5100
           CALL CD_GET_VAR_ATT_INFO (dset, varid, attid, 
     .                attname, attype, attlen, attoutflag, status )
	   IF (status.EQ.atom_not_found) GOTO 5100
	   got_it = NC_GET_ATTRIB_STRING (dset, varid, attname, 
     .                                do_warn, 'global', max_slen,  
     .                                attlen, attoutflag, axname)
	   axis = TM_GET_LINENUM(axname)
	   IF (axis .GT. max_lines) CALL TM_DEALLO_DYN_LINE(axis)

	   CALL CD_GET_VAR_ATT_ID (dset, varid, '_lag_f_ax', attid, status)
	   IF (status.EQ.atom_not_found) GOTO 5100
           CALL CD_GET_VAR_ATT_INFO (dset, varid, attid, 
     .                attname, attype, attlen, attoutflag, status )
	   IF (status.EQ.atom_not_found) GOTO 5100
	   got_it = NC_GET_ATTRIB_STRING (dset, varid, attname, 
     .                                do_warn, 'global', max_slen, 
     .                                attlen, attoutflag, axname)
	   axis = TM_GET_LINENUM(axname)
	   IF (axis .GT. max_lines) CALL TM_DEALLO_DYN_LINE(axis)
	ENDIF

* really close and cancel the data set
	CALL TM_CLOSE_SET( dset, status )
	IF ( status .NE. merr_ok ) RETURN

* ancient legacy stuff: was this a climatological air temp field ?
	DO 400 iclim = 1, max_airt_clim
 	   IF ( dp_airt_dset(iclim) .EQ. dset ) THEN
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
     .                                   CALL ERASE_DSET( dset )
 600	CONTINUE

        RETURN

* error exits
 5000	RETURN
 5100	CALL ERRMSG(ferr_internal, status, 'cancel_data_set', *5000 )

	END
