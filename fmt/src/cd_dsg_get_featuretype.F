	SUBROUTINE CD_DSG_GET_FEATURETYPE (cdfid, ftype, orientation, dsg_ragged, do_warn)


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
* Find the featureType attribute and return the feature-type information
* for the dataset
* V751 8/18 *acm* don't issue a note if the file has featureType="grid"
* v76  2/20 *acm* add handling of Point feature-type
* V7.6 *acm* 6/20 Issue 1980: When possible open invalid DSG datasets anyway with NOTES.


        include 'netcdf.inc'
        include 'tmap_errors.parm'
c	include 'tmap_dims.parm'
#include "gt_lib.parm"
#include "tmap_dset.parm"
	include 'xio.cmn_text'


	INTEGER cdfid, ftype, orientation
	LOGICAL	dsg_ragged, do_warn

	LOGICAL got_it, CD_GET_ATTRIB
	INTEGER STR_CASE_BLIND_COMPARE, varid, attlen, maxlen
	CHARACTER buff*32

	ftype = pfeatureType_Point
	orientation = no_dim
	dsg_ragged = .FALSE.
	maxlen = 32

	varid = NF_GLOBAL
	got_it = CD_GET_ATTRIB ( cdfid, varid, 'featureType', do_warn, '.',
     .                           buff, attlen, maxlen )
	IF (.NOT. got_it) RETURN

	IF (STR_CASE_BLIND_COMPARE(buff,"timeseries") .EQ. str_match) THEN
	   ftype = pfeatureType_TimeSeries
	   orientation = t_dim		! T axis
	   dsg_ragged = .TRUE.

	ELSEIF (STR_CASE_BLIND_COMPARE(buff,"profile") .EQ. str_match) THEN
	   ftype = pfeatureType_Profile
	   orientation = z_dim		! Z axis
	   dsg_ragged = .TRUE.

	ELSEIF (STR_CASE_BLIND_COMPARE(buff,"trajectory") .EQ. str_match) THEN
	   ftype = pfeatureType_Trajectory
	   orientation = x_dim		! X axis (arbitrarily)
	   dsg_ragged = .TRUE.


	ELSEIF (STR_CASE_BLIND_COMPARE(buff,"point") .EQ. str_match) THEN
	   ftype = pfeatureType_Point
	   orientation = x_dim		! X axis (arbitrarily)
	   dsg_ragged = .TRUE.

	ELSEIF (STR_CASE_BLIND_COMPARE(buff,"pfeatureType_TrajectoryProfile") .EQ. str_match) THEN
	   ftype = pfeatureType_TrajectoryProfile
	   orientation = y_dim		! Y axis (arbitrarily)
	   dsg_ragged = .TRUE.

	ELSEIF (STR_CASE_BLIND_COMPARE(buff,"pfeatureType_TimeseriesProfile") .EQ. str_match) THEN
	   ftype = pfeatureType_TimeseriesProfile
	   orientation = f_dim		! F axis (arbitrarily)
	   dsg_ragged = .TRUE.

	ELSE
	   dsg_ragged = .FALSE.
	   IF (STR_CASE_BLIND_COMPARE(buff,"grid") .NE. str_match .AND. do_warn)
     .     CALL TM_NOTE("Ignoring non-DSG FeatureType: "//buff , lunit_errors)

	ENDIF

	RETURN
	END
