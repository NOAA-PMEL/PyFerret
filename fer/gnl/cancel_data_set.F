	RECURSIVE SUBROUTINE CANCEL_DATA_SET( dset, status )

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
* close a dataset
* remove from memory any cached data that may depend upon this dataset
* if the dataset is a part of an E or F aggregations refuse to close
* unless all datasets are being closed (slash/all)

* programmer - steve hankin
* NOAA/PMEL, Seattle, WA - Tropical Modeling and Analysis Program
* written for VAX computer under VMS operating system
*
* V695 *sh* 1/16 - bulk of logic moved to CANCEL_ONE_DATA_SET 
* V698  3/16 *acm* Remove unneeded xdset_parms.cmn

* calling arguments
*  dset      -  (input) dataset number to be closed
*  status    - (output) return status

	include 'tmap_dims.parm'
#	include "tmap_dset.parm"
	include 'tmap_errors.parm'
	include	'ferret.parm'
	include 'errmsg.parm'
	include 'xprog_state.cmn'
	include 'xdset_info.cmn_text'
	external xdset_info_data
	include 'xrisc.cmn'
	include	'xtm_grid.cmn_text'
	external xgt_grid_data

* calling argument declarations:
	INTEGER	dset, status

* local variable declarations:
	LOGICAL	IS_AGG_MEMBER, is_member, more
	INTEGER	one_set_status, parent, imemb, nmemb, membset
 
* initialize
	status = ferr_ok
	IF (ds_name(dset) .EQ. char_init1024) RETURN  ! already deleted
	more = .TRUE.

* is the requested dataset part of an aggregation?
* If so, this is an implicit request to (recursively) cancel the parent, too
* Remember that E aggs can be members of F aggs and visa versa
	DO WHILE (more)
	   IF ( IS_AGG_MEMBER(dset, 1, parent, more) ) THEN
	      CALL WARN('Dataset used in aggregation.  '//
     .		        'Canceling aggregation, '//ds_name(parent))
	      CALL CANCEL_DATA_SET(parent, status)
	      IF (status .NE. ferr_ok) GOTO 5000
	   ENDIF
	ENDDO

* delete all user defined data from memory for similar reasons
* (note: this is unnecessarily strict - could be refined in the future)
	CALL PURGE_ALL_UVARS

* if this dataset is an aggregation, cancel hidden member datasets, too
* unless they are also members in some other aggregation
	IF (ds_type(dset).EQ.'ENS' .OR. ds_type(dset).EQ.'FCT') THEN

* ... loop over aggregation members
	   CALL CD_GET_AGG_DSET_INFO (dset, nmemb, status)
	   IF (status .NE. ferr_ok) CALL WARN('crptn: isit_agg_member')
	   DO imemb = 1, nmemb
	      CALL CD_GET_AGG_DSET_MEMBER (dset, imemb, membset, status)
	      IF (status .NE. ferr_ok)  CALL WARN('crptn: cancel_data_set')
	      IF (ds_name(membset) .EQ. char_init1024) CYCLE ! already deleted?

* ... if not a member of some other aggregation: cancel the hidden member
	      is_member = IS_AGG_MEMBER(membset, 1, parent, more)
	      IF ( is_member .AND. .NOT.more ) THEN
	         IF (ds_hide(membset)) CALL CANCEL_ONE_DATA_SET
     .							(membset, status)
	         IF (status .NE. ferr_ok) GOTO 5000
	      ENDIF
	   ENDDO
	ENDIF

* finally cancel the requested dataset
	CALL CANCEL_ONE_DATA_SET(dset, status)
	IF (status .NE. ferr_ok) GOTO 5000

* success
        RETURN

* error exits
 5000	RETURN
	END
