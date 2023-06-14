      SUBROUTINE CD_ADD_COORD_VAR (dset, varname, varid, units, 
     .                       type, coordvar, bad, status)

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
* Add a coordinate variable to the linked list structure.  dset = -2 will be the 
* pseudo-dset containing attribute info for user-defined axes (coordinate variables).

* Programmer Ansley Manke
* NOAA/PMEL, Seattle, WA - TMAP
*
* V680 5/12 *acm* 6D Ferret, changes for DEFINE DATA/AGGREGATE

* include files

      INCLUDE "tmap_dtypes.parm"
      INCLUDE "netcdf.inc"

* Argument declarations

      CHARACTER*(*) varname, units
      INTEGER dset, varid, type, coordvar, status
      REAL bad

* Internal declarations

      INTEGER TM_LENSTR1, TM_LENSTR, NCF_ADD_VAR, NCF_ADD_COORD_VAR, 
     .        slen, flen, dset_num
      REAL*8 badd
      CHARACTER*2048 buff

      PARAMETER (flen=2048)
      INTEGER*1 vhol(flen)      ! c-type Hollerith string buffer
      INTEGER*1 uhol(flen)

* translate strings for C routine

      buff = varname
      slen = TM_LENSTR1( buff )
      CALL TM_FTOC_STRNG( buff(1:slen), vhol, flen )

      buff = units
      slen = TM_LENSTR( buff )
      CALL TM_FTOC_STRNG( buff(1:slen), uhol, flen )

* attribute structure has missing data flag as double precision

      badd = bad

      dset_num = dset
      status = NCF_ADD_COORD_VAR( dset_num, varid, type, coordvar,
     .                      vhol, uhol, badd)

      RETURN
      END
