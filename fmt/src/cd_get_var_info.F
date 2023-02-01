      SUBROUTINE CD_GET_VAR_INFO (dset, ivar, vname, vartype, 
     .            nvdims, vdims, nvatts, coordvar, outflag, cdfstat)

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

* See also these routines which get subsets of these. 
* All of the below calls take either a variable id or variable name. 
* If the varid is < 0, it looks at the variable name and gets the varid.
*
*  The variable dimension info, 
*       SUBROUTINE CD_GET_VAR_DIMS_INFO (dset, ivar, vname, nvdims, 
*     .        vdims, status)
*
*  The logical flag true if it is a coordinate variable
*       LOGICAL FUNCTION CD_ISIT_COORDVAR (dset, ivar, vname, status)
*
*  The number of attributes the variable hase
*      SUBROUTINE CD_GET_VAR_NATTS (dset, ivar, vname, nvatts, status)
*
*  The flag, output-all-attributes
*      SUBROUTINE CD_GET_VAR_OUTFLAG (dset, ivar, vname, outflag, status)
*
*  The variable type, netCDF type 
*     SUBROUTINE CD_GET_VAR_TYPE (dset, varid, vname, vtype, status)
*
*  The variable name as it appears in the input file or variable definition,
*  including the original upper- or lower-case spelling
*      SUBROUTINE CD_GET_VAR_VARNAME (dset, varid, vname, status)
*
*
* NOTE
* 5/21/2020 
* Note that data types greater than NF_DOUBLE have been added.
* The numeric ones of these seem to work fine for us, but NF_STRING
* does not. CD_SCAN_VARS and CD_DSG_SCAN_VARS have traps to issue
* an error if datasets have variables with type.

* Programmer Ansley Manke
* NOAA/PMEL, Seattle, WA - TMAP

* V600 6/05 *acm* Attribute control.
* V602  2/07 *acm* Fix bug 1492, changing attributes of coordinate variables 
* V674 2/12 *acm* 6D Ferret: use nferdims rather than 4 for dimension indexing
* V680 5/12 *acm* 6D Ferret, changes for DEFINE DATA/AGGREGATE
* V698 *sh* 1/16 -- added status check/warning, since not checked by callers
* V74+ 4/18 *acm* Getting DSG grids

* Argument declarations

      CHARACTER*(*) vname
      INTEGER dset, ivar, vartype, nvdims, vdims(*), nvatts, 
     .        outflag, cdfstat, dset_num
      LOGICAL coordvar

* Internal declarations

#include "gt_lib.parm"
      INCLUDE 'netcdf.inc'
      INCLUDE 'tmap_errors.parm'
      INCLUDE 'tmap_dims.parm'

      INTEGER TM_LENSTR1, NCF_INQ_VAR, slen, flen, rev(8), i, ndx, istat
      PARAMETER (flen=512)      ! size of variable name string
      INTEGER*1 fhol(flen)      ! c-type Hollerith string buffer

      nvdims = 0
      nvatts = 0
      coordvar = .FALSE.
      cdfstat = atom_not_found

      DO i = 1, nferdims
         vdims(i) = line_0
      ENDDO

      IF (dset .EQ. unspecified_int4) RETURN
      IF (ivar .LT. 0) RETURN

      dset_num = dset
      cdfstat = NCF_INQ_VAR(dset_num, ivar, fhol, slen, vartype, 
     .            nvdims, nvatts, coordvar, outflag, rev)
      IF (cdfstat .NE. pcd_ferr_ok) GOTO 5000  ! not found
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
