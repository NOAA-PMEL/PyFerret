	LOGICAL FUNCTION IS_AGG_MEMBER( dset, parent, nparent )

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
* is the indicated dataset part of some other aggregated dataset
!**** NOTE:  THIS ROUTINE (and many others) SHOULD USE THE dset
!****      LIST STRUCTURE RATHER THAN BRUTE-FORCE LOOPS THROUGH ALL DATASETS

* calling arguments
*   dset    - the dataset to be checked to see if it is used in an agg (input)
*   parent  - the last agg that owns it (or unspecified_int4 if none) (output)
*   nparent - the number of aggs for which this dataset is a member

* programmer - steve hankin
* V695 *sh* 1/16

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
	INTEGER	dset, parent, nparent

* local variable declarations:
	INTEGER	nmemb, imemb, membset, iset, status


* initialize
	nparent = 0

* check all of the aggregations currently open
	DO iset = 1, maxdsets
	   IF ( ds_name(iset) .EQ. char_init1024) CYCLE
	   IF (iset .EQ. dset) CYCLE
	   IF ( ds_type(iset).NE.'ENS' .AND. ds_type(iset).NE.'FCT') CYCLE 

* Get number of aggregation members
	   CALL CD_GET_AGG_DSET_INFO (iset, nmemb, status)
	   IF (status .NE. ferr_ok) CALL WARN('crptn: isit_agg_member')

* Look through the aggregation members to see if we are one of them
	   DO imemb = 1, nmemb
	      CALL CD_GET_AGG_DSET_MEMBER (iset, imemb, membset, status)
	      IF (status .NE. ferr_ok)  CALL WARN('crptn: isit_agg_member')
	      IF (membset .EQ. dset) THEN
	         nparent = nparent + 1
	         parent = iset   ! the last one found will be returned
	         IS_AGG_MEMBER = .TRUE.
	      ENDIF
	   ENDDO
	ENDDO
	IF (nparent .GT. 0) RETURN

* not a member of any aggregation
	parent = unspecified_int4
	IS_AGG_MEMBER = .FALSE.
	RETURN
	END
