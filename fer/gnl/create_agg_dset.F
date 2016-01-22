	SUBROUTINE CREATE_AGG_DSET (agg_dset, dname, dtitle,
     .               nsets, dset1, agg_dim, iline, status)
*
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
* Programmer Ansley Manke
* NOAA/PMEL, Seattle, WA - TMAP 
*
* V680 5/12 *acm* 6D Ferret, changes for DEFINE DATA/AGGREGATE
* V685 *acm*  3/13 For Ensemble aggregations, call the ds_type 'ENS'
* V695 *acm*  2/15 For Forecast aggregations, call the ds_type 'FCT'
* V695+ *sh*  11/15 Correct bugs in setting of aggregate dset title

* calling argument
*   agg_dset	- dataset number of the aggregation (input)
*   dname	- name of the aggregated dataset (input)
*   dtitle	- optional title string (input)
*   nsets	- number of datasets in the aggregation (input)
*   dset1	- dataset number of first (pattern) dataset in agg (input)
*   agg_dim	- idim number (5 or 6) of aggregation dimension (input)
*   iline	- ??
*   status	- return status (output)

      IMPLICIT NONE
        include 'tmap_errors.parm'
#include "tmap_dset.parm"
        include 'tmap_dims.parm'
	include	'ferret.parm'
	include 'xdset_info.cmn_text'
	external xdset_info_data
	include 'xdset_info.cd_equiv'
        include 'xprog_state.cmn'
	include 'xtm_grid.cmn_text'  
	include 'errmsg.parm'
	include 'netcdf.inc'
	include 'xvariables.cmn'
	include 'xtext_info.cmn'

* calling argument declarations:
	CHARACTER*(*)	dname, dtitle
	INTEGER		agg_dset, nsets, dset1, agg_dim, iline, status

* local variable declarations:
	INTEGER TM_LENSTR, TM_LENSTR1, ALLO_MANAGED_AXIS,
     .		type, coordvar, varid, big_cx, nlen
	CHARACTER LEFINT*20, ntext*20
	REAL	bad_flag

	CALL CD_INIT_AGG_DSET (agg_dset, dname, status)

	IF (status .NE. ferr_ok) GOTO 5300 

* Set the dataset type and title (can add other directions)
	IF (agg_dim .EQ. e_dim ) ds_type(agg_dset) = 'ENS'
	IF (agg_dim .EQ. f_dim ) ds_type(agg_dset) = 'FCT'

	ds_name(agg_dset) = dname
	ds_des_name(agg_dset) = dname

	IF (dtitle(1:2).EQ.char_init .OR. TM_LENSTR(dtitle).EQ.0) THEN
	   ntext = LEFINT(nsets, nlen)
	   IF (agg_dim.EQ.e_dim) THEN
	      ds_title(agg_dset) = "Ensemble"
	   ELSE
	      ds_title(agg_dset) = "Forecast"
	   ENDIF
	   ds_title(agg_dset) = ds_title(agg_dset)(:8)//' series of '
     .		//ntext(:nlen)//' datasets patterned on '//
     .		ds_name(dset1)
	ELSE
	   ds_title(agg_dset) = dtitle
	ENDIF
	ds_mod_title(agg_dset) = ' '
 
* create the aggregate axis

	CALL CREATE_AGG_AXIS (nsets, agg_dim, iline, status)
	IF (status .NE. ferr_ok) GOTO 5300 

	RETURN
	
 5000   RETURN	
 5300   status = ferr_TMAP_error
	RETURN

	END
