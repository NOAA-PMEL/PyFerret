	SUBROUTINE CD_AXNAME_TO_WRITE (cdfid, idim, name, nlen, cdfstat)

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
* NOAA/PMEL, Seattle, WA - SDIG group
* revision 0.0 - 9/16
* For Ticket 2459

* Given cdfid for an open netCDF file we are writing, the dimension direction,
* and an axis name, return the name for writing, perhaps with a direction 
* appended. If there is already an axis with the same name but a different 
* direction. nlen on input is the length of name, on output
* it will be the length of the resulting name.

        include 'netcdf.inc'
	include 'tmap_dims.parm'
	include 'xtext_info.cmn'  ! with ww_dim_name

* Declare netcdf function names. 
* The ones without trailing underscore are declared in netcdf.inc
#include "netcdf_declareUnderscore.inc"

	INTEGER cdfstat, cdfid, idim, nlen
	CHARACTER*(*) name

	INTEGER   cvarid, local_status
	CHARACTER buff*5, axis_dir*1


#ifdef usingDODSf2cUnderscore
	local_status = NF_INQ_VARID_( cdfid, name(:nlen), cvarid)
#else
	local_status = NF_INQ_VARID( cdfid, name(:nlen), cvarid)
#endif

	buff = ' '	 
	IF (local_status .EQ. NF_NOERR) THEN
#ifdef usingDODSf2cUnderscore
           local_status = NF_GET_ATT_TEXT_ (cdfid, cvarid, 'axis', buff)
#else
           local_status = NF_GET_ATT_TEXT (cdfid, cvarid, 'axis', buff)
#endif
	ENDIF

* If we change the axis name, mark status as 'dimension not found' in the
* file, then Ferret will go on and write the axis with the new name.

	IF (local_status .EQ. NF_NOERR) THEN
	    CALL STR_UPCASE(axis_dir, buff(:1))
	    IF (axis_dir .NE. ww_dim_name(idim))  THEN
	       name = name(:nlen)//ww_dim_name(idim)
	       nlen = nlen+1
	       cdfstat = NF_EBADDIM
	    ENDIF
	ENDIF

	RETURN
	END
