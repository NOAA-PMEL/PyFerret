      SUBROUTINE CD_GET_ID_VARNAME (dset, varid, vname, status)

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
* given dset and variable id return variable name

* Programmer Ansley Manke
* NOAA/PMEL, Seattle, WA - TMAP

* V7   *acm* 8/16

* include files
#include "gt_lib.parm"
      INCLUDE "tmap_errors.parm"

* Argument declarations

      CHARACTER*(*) vname
      INTEGER dset, varid, status

* Internal declarations

      INCLUDE 'netcdf.inc'

      INTEGER TM_LENSTR1, NCF_GET_VAR_NAME,
     .        slen, flen, dset_num
      CHARACTER*512 errbuff
      PARAMETER (flen=512)      ! size of variable name string
      INTEGER*1 fhol(flen)      ! c-type Hollerith string buffer

      status = atom_not_found

      dset_num = dset
      IF (dset_num .LT. -2) dset_num = -2

      status = NCF_GET_VAR_NAME(dset_num, varid, fhol, slen) 
      IF (status .NE. merr_ok) GOTO 5000

* Return vname in original upper/lower case

      slen = MIN(slen, flen)
      CALL TM_CTOF_STRNG ( fhol, vname, slen )


      RETURN

* error exit
 5000 IF (status .NE. atom_not_found) THEN
	CALL CD_TRANSLATE_ERROR (status, errbuff)
	slen = TM_LENSTR1(errbuff)

        CALL TM_ERRMSG
     .     ( merr_badfileatt, status, 'CD_NF_GET_VARID', dset,
     .        no_stepfile, errbuff, no_errstring, *5000 )

      ENDIF
      RETURN
      END
