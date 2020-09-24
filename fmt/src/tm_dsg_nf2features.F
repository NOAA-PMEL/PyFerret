	INTEGER FUNCTION TM_DSG_NF2FEATURES(dset)

*
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
*
* return the number of second-level features in this DSG grid
* e.g. number of stations for timeseries profile or number of 
* trajectories for trajectoryprofile

* V761 *acm* 8/20


        include 'netcdf.inc'
	include 'tmap_errors.parm'
	include 'tmap_dims.parm'
	include 'errmsg.parm'
#include "tmap_dset.parm"
	include 'xdset_info.cmn_text'
	include 'xdset_info.cd_equiv'
	external xdset_info_data

* argument declarations
	INTEGER dset

* internal variable declarations
	INTEGER ndims, vdims(2), f2dim, npts, status
	CHARACTER*32 name

	TM_DSG_NF2FEATURES = 0

	IF (dsg_ftrset_var(dset) .EQ. int4_init) GOTO 5500

* If they said use/feature=profile then n2features = 0

	IF (dsg_orientation(dset).NE.pfeatureType_TrajectoryProfile .AND.
     .      dsg_orientation(dset).NE.pfeatureType_TimeseriesProfile) RETURN

	CALL CD_GET_VAR_DIMS_INFO ( dset, cd_varid(dsg_ftrset_var(dset)), 
     .        name, ndims, vdims, status)
     
	IF (status .NE. merr_ok) GOTO 5500
	f2dim = vdims(1)
	IF (ndims .EQ. 2) f2dim = vdims(2)
	
	CALL CD_GET_DS_DIMS (dset, f2dim, name, npts, status)
	IF (status .NE. merr_ok) GOTO 5500

	TM_DSG_NF2FEATURES = npts

* success
        RETURN
 5500	RETURN

        END








