	SUBROUTINE GET_DSET_BAD( dset, varname, bad )

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
* Return the bad-flag for the variable. For variables in netCDF datasets, 
* with scale and offset, the internal bad-flag may not match what is in
* the original file.  For `var,return=bad`, go get the original flag.

* programmer - Ansley Manke
* NOAA/PMEL, Seattle, WA -SDIG
*
* revision 0.0 - 2/18
* For Issue # 1861, handling missing- and _FillValue with scale and offset
* this is for evaluating `var,RETURN=bad`


	include 'tmap_dims.parm'
#	include "tmap_dset.parm"
	include 'xdset_info.cmn_text'
	external xdset_info_data
	include 'ferret.parm'

* calling argument declarations:
	INTEGER		dset
	REAL		bad
	CHARACTER*(*)   varname

	LOGICAL		NC_GET_ATTRIB_DP, got_it
	INTEGER		varid, istat, attid, attlen, attoutflag
	REAL		miss_in
	CHARACTER*128   attname


	CALL CD_GET_VAR_ID (dset, varname, varid, istat)

	IF ( ds_type(dset).EQ.'CDF' .OR. ds_type(dset).EQ.'ECDF' 
     .          .OR. ds_type(dset).EQ.' MC' .OR. ds_type(dset).EQ.'ENS' 
     .          .OR. ds_type(dset).EQ.'FCT' .OR. ds_type(dset).EQ.'UNI') THEN

	   attname = 'missing_value'
	   got_it = NC_GET_ATTRIB_DP 
     .                (dset, varid, attname, .FALSE., varname, attlen, 
     .                attoutflag, miss_in)
	   IF (.NOT. got_it) THEN
	      attname = '_FillValue'
	      got_it = NC_GET_ATTRIB_DP 
     .                (dset, varid, attname, .FALSE., varname, attlen, 
     .                attoutflag, miss_in)
	   ENDIF
	   IF (got_it) bad = miss_in
	
	ENDIF



 


	RETURN
	END
