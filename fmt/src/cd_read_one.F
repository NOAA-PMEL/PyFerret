	SUBROUTINE CD_READ_ONE (dset, ivar, index, dtype, val, buff, slen, status)

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
*
* Reads a single value or string from a netCDF variable, must be a 1-D var.

* written 8/18 (e.g. getting one value at a time from a DSG feature-id variable)
*

* include files
        include 'netcdf.inc'
	include 'tmap_errors.parm'
	include 'tmap_dims.parm'		! data set dimension
#include "gt_lib.parm"
	include 'xdset_info.cmn_text'	! data set info common
	external xdset_info_data
	include 'xstep_files.cmn_text'	! step file common
	external xstep_files_data
        include 'xdset_info.cd_equiv'


* arguments:
*	dset        - internal data set # 
*	ivar        - position in common of variable information
*	index       - index of value of 1-d variable to read
*	val         - numeric value for num data
*	buff        - String returned for string data
*	dtype       - data type of the variable
*	slen        - length of string if string var
*	status      - error status upon completion (output)
*
	INTEGER		dset, ivar, i, index, dtype, slen, status
	REAL     	val
	CHARACTER*(*)	buff


* local definitions

        INTEGER	cdfid, varid, nvdims, vdims(nferdims)
        CHARACTER*128 vname

* initialize
        cdfid = sf_lunit(dset)
        varid = cd_varid(ivar)

        DO i = 1,maxvars
           IF (ds_var_setnum(i).EQ.dset .AND. cd_varid(i).EQ.ivar) THEN 
	      varid = cd_varid(i)
	      EXIT
	   ENDIF
	ENDDO

* Get variable type

	CALL CD_GET_VAR_TYPE (dset, varid, vname, dtype, status)

* This routine is only for 1-D variables. Check that.

        CALL CD_GET_VAR_DIMS_INFO (dset, varid, vname, nvdims, 
     .        vdims, status)

	IF (dtype .EQ. NF_CHAR) nvdims = nvdims-1
	IF (nvdims .NE. 1) GOTO 5100


	IF (dtype .EQ. NF_CHAR) THEN
	   CALL CD_RD_STR_1(cdfid, varid, index, vname, buff, slen, status)

	ELSE
	   CALL CD_RD_R8_1(cdfid, varid, index, dtype, vname, val, status)

	ENDIF
	IF (status .NE. merr_ok) GOTO 5200

 5000	RETURN

 5100	CALL ERRMSG( merr_dsg_grid, status, 
     .     'Getting Feature IDs. Feature var not 1-D?',  *5000 )
 5200   CALL TM_ERRMSG
     .     ( status+pcdferr, status, 'CD_READ_one', cdfid, ivar,
     .       no_errstring, no_errstring, *5000 )

	END
