      SUBROUTINE CD_GET_VAR_TYPE (dset, varid, vname, vtype, status)

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
* Read the linked list structure for netcdf variable information: 
* given dset and variable name OR variable ID return variable type.
* If varid is 0 or -999, use vname
*
*
* NOTE
* 5/21/2020 
* Note that data types greater than NF_DOUBLE have been added.
* The numeric ones of these seem to work fine for us, but NF_STRING
* does not. CD_SCAN_VARS and CD_DSG_SCAN_VARS have traps to issue
* an error if datasets have variables with type.

* Programmer Ansley Manke
* NOAA/PMEL, Seattle, WA - 
* 5/2017

* include files
#include "gt_lib.parm"
      INCLUDE "tmap_errors.parm"
      INCLUDE 'netcdf.inc'

* Argument declarations

      CHARACTER*(*) vname
      INTEGER dset, varid, vtype, status
   
* Internal declarations
      INTEGER NCF_GET_VAR_TYPE, TM_LENSTR1
      INTEGER slen, var_id

      status = merr_ok

      var_id = varid
      IF (varid.EQ.0 .OR. varid.EQ.unspecified_int4) THEN
        slen = TM_LENSTR1(vname)
	CALL CD_GET_VAR_ID (dset, vname(:slen), var_id, status)
      ENDIF

      status = NCF_GET_VAR_TYPE(dset, var_id, vtype)
      
      RETURN
      END
