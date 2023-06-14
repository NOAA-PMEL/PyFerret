      SUBROUTINE CD_GET_VAR_DIMS_INFO (dset, ivar, vname, nvdims, 
     .        vdims, status)

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
* Read the linked list structure for netcdf variable information: given
* dset and ivar return variable name (in its original upper/lowercase 
* form), type, dimensions, number of attributes

* Programmer Ansley Manke
* NOAA/PMEL, Seattle, WA - TMAP
* V74+ 4/18 *acm* Getting DSG grids

* Argument declarations

      CHARACTER*(*) vname
      INTEGER dset, ivar, nvdims, vdims(*), status

* Internal declarations

#include "gt_lib.parm"
      INCLUDE 'netcdf.inc'
      INCLUDE 'tmap_errors.parm'

      INTEGER TM_LENSTR1, NCF_INQ_VAR, slen, flen, rev(8), i, ndx, 
     .	      vartype, nvatts, outflag, dset_num
      LOGICAL coordvar

      PARAMETER (flen=512)      ! size of variable name string
      INTEGER*1 fhol(flen)      ! c-type Hollerith string buffer

      nvdims = 0
      status = atom_not_found

      IF (dset .EQ. unspecified_int4) RETURN
      
* If the varid is not set, find it from the variable name
      IF (ivar.LE.0 .OR. ivar.EQ.unspecified_int4) THEN
        slen = TM_LENSTR1(vname)
        CALL CD_GET_VAR_ID (dset, vname(:slen), ivar, status)
        IF (ivar.LT.0 .OR. status.NE.pcd_ferr_ok) RETURN
      ENDIF

      IF (ivar .LT. 0) RETURN

      dset_num = dset
      status = NCF_INQ_VAR(dset_num, ivar, fhol, slen, vartype, 
     .            nvdims, nvatts, coordvar, outflag, rev)
      IF (status .NE. pcd_ferr_ok) GOTO 5000  ! not found
      slen = MIN(slen, flen)

* Return vname in original upper/lower case
      CALL TM_CTOF_STRNG ( fhol, vname, slen )

* These dims are 0,1,2,3. For Fortran want 1,2,3,4, so add 1
* Also, the dims are in C order t,z,y,x. Reverse the order, for 
* opposite array ordering of Fortran vs c

      DO 100 i = 1, nvdims
	  ndx = nvdims - i + 1
	  vdims(i) = rev(ndx) + 1
 100  CONTINUE   

5000  RETURN
      END
