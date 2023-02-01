      SUBROUTINE CD_PUT_NEW_ATTR_DP (dset, varid, attname, attype, 
     .   attlen, attoutflag, string, vals, status )

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
* Add an attribute to a variable in the linked list structure for netcdf 
* variable information: given dset, varid, attribute specification

* Programmer Ansley Manke
* NOAA/PMEL, Seattle, WA - TMAP

* V601 11/06 *acm* Attribute control, adding double precision attribute value
* V602  2/07 *acm* Fix bug 1492, changing attributes of coordinate variables 
* V65  *acm* 1/10  Change all netcdf2-style calls and parameter names to 
*                  netcdf3-style calls, which are also consistent with netcdf4.
* V65  *acm* 1/10  Make sure parameters values are consistent with 
*                              whats in netcdf.inc from netcdf4.
* V6.74 3/12 *acm* cleanup ifdefs and unnecessary include files

* Argument declarations

      CHARACTER*(*) attname, string
      INTEGER dset, varid, attype, attlen, attoutflag, status
      REAL*8 vals(*)

* Internal declarations

      include 'netcdf.inc'
      include 'tmap_errors.parm'
      include 'xrisc_buff.cmn'

      LOGICAL cv
      INTEGER TM_LENSTR1, NCF_ADD_VAR_NUM_ATT_DP, NCF_ADD_VAR_STR_ATT,
     .        flen, slen, wlen, vdims(8), vartype, nvdims, nvatts, 
     .        all_outflag, dset_num

      CHARACTER*128 vname

      PARAMETER (flen=size_rbuff)      ! size of attribute string
      INTEGER*1 fhol(flen), ghol(flen)      ! c-type Hollerith string buffer


      slen = TM_LENSTR1( attname )
      CALL TM_FTOC_STRNG( attname(1:slen), fhol, flen )

      dset_num = dset
      IF (dset_num .LT. -2) dset_num = -2

      IF (attype .NE. NF_CHAR) THEN

         status = NCF_ADD_VAR_NUM_ATT_DP(dset_num, varid, fhol, attype, 
     .           attlen, attoutflag, vals)
         IF (status .LT. 0) GOTO 5300

      ELSE
         slen = TM_LENSTR1( string)
         CALL TM_FTOC_STRNG( string(1:slen), ghol, flen )
         
         status = NCF_ADD_VAR_STR_ATT(dset_num, varid, fhol, attype, 
     .           slen, attoutflag, ghol)
         IF (status .LT. 0) GOTO 5300

      ENDIF

5000  RETURN

! Error
 5300 CALL CD_GET_VAR_INFO (dset_num, varid, vname, vartype, nvdims, 
     .   vdims, nvatts, cv, all_outflag, status)
      slen = TM_LENSTR1( attname )
      wlen = TM_LENSTR1( vname )
      risc_buff(1:slen) = attname
      risc_buff(slen+1:slen+14) = ' for variable '
      risc_buff(slen+15:slen+15+wlen) = vname
      CALL TM_ERRMSG (merr_attalready, status, 'CD_PUT_NEW_ATTR',
     .           dset_num, no_stepfile, 
     .           risc_buff(1:slen+15+wlen),
     .           no_errstring, *5000)
      END
