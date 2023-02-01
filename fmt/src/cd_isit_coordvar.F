       LOGICAL FUNCTION CD_ISIT_COORDVAR (dset, ivar, vname, status)

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
* given dset and variable name OR variable ID return True if it is a 
* coordinate variable.
* If varid is 0 or -999, use vname

* Programmer Ansley Manke
* NOAA/PMEL, Seattle, WA - TMAP


#include "gt_lib.parm"
      INCLUDE 'netcdf.inc'
      INCLUDE 'tmap_errors.parm'

* Argument declarations

      CHARACTER*(*) vname
      INTEGER dset, ivar, status

* Internal declarations

      INTEGER TM_LENSTR1, NCF_INQ_VAR, slen, flen, rev(8), 
     .        vartype, nvdims, nvatts, outflag, dset_num
      LOGICAL coordvar

      PARAMETER (flen=512)      ! size of variable name string
      INTEGER*1 fhol(flen)      ! c-type Hollerith string buffer

      CD_ISIT_COORDVAR = .FALSE.

      coordvar = .FALSE.
      status = merr_ok

      IF (dset .EQ. unspecified_int4) RETURN
      
* If the varid is not set, find it from the variable name
      IF (ivar.LE.0 .OR. ivar.EQ.unspecified_int4) THEN
         slen = TM_LENSTR1(vname)
	 CALL CD_GET_VAR_ID (dset, vname(:slen), ivar, status)
      ENDIF

      IF (ivar.LT.0 .OR. status.NE.pcd_ferr_ok) RETURN

      dset_num = dset
      status = NCF_INQ_VAR(dset_num, ivar, fhol, slen, vartype, 
     .            nvdims, nvatts, coordvar, outflag, rev)
      IF (status .NE. pcd_ferr_ok) GOTO 5000  ! not found

      CD_ISIT_COORDVAR = coordvar

5000  RETURN
      END
