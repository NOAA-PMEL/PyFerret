      SUBROUTINE CD_SET_ATT_FLAG (dset, varid, attname, iflag, status )

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
* Set the attribute output flag to iflag (1 or 0)

* Programmer Ansley Manke
* NOAA/PMEL, Seattle, WA - TMAP

* V600 12/05 *acm* Attribute control.
* V602  2/07 *acm* Fix bug 1492, changing attributes of coordinate variables 
* V65  *acm* 1/10  Make sure parameters values are consistent with 
*                              whats in netcdf.inc from netcdf4.
* V6.74 3/12 *acm* cleanup ifdefs and unnecessary include files

* Argument declarations

      CHARACTER*(*) attname
      INTEGER dset, varid, iflag, status

* Internal declarations

      include 'netcdf.inc'
      include 'tmap_errors.parm'
      include 'xrisc_buff.cmn'

      LOGICAL cv
      INTEGER TM_LENSTR, NCF_SET_ATT_FLAG, NCF_SET_VAR_OUT_FLAG, 
     .        flen, slen, wlen, 
     .        i, vtype, nvdims, nvatts, vdims(8), all_outflag, dset_num
      CHARACTER*128 vname

      PARAMETER (flen=size_rbuff)      ! size of attribute string
      INTEGER*1 fhol(flen)      ! c-type Hollerith string bufferen)      ! c-type Hollerith string buffer

      dset_num = dset
      IF (dset_num .LT. -2) dset_num = -2

* set output flag for this variable and attribute

      slen = TM_LENSTR( attname )
      IF (slen .GT. 0) THEN
      
         CALL TM_FTOC_STRNG( attname(1:slen), fhol, flen )
         status = NCF_SET_ATT_FLAG( dset_num, varid, fhol, iflag)

         IF (status .LT. 0) GOTO 5300

      ELSE     ! SET ATT/OUT=all or SET ATT/OUT=default

         status = NCF_SET_VAR_OUT_FLAG( dset_num, varid, iflag)

c* set flag for all attributes of this variable
c
c         CALL CD_GET_VAR_INFO (dset, varid, vname, vtype,
c     .                              nvdims, vdims, nvatts, cv, 
c     .                              all_outflag, status)
c         DO 100 i = 1, nvatts
c            CALL CD_GET_VAR_ATT_NAME (dset, varid, i, attname, status )
c
c            slen = TM_LENSTR( attname )
c            CALL TM_FTOC_STRNG( attname(1:slen), fhol, flen )
c         
c            status = NCF_SET_ATT_FLAG( dset, varid, fhol, iflag)
c            IF (status .LT. 0) GOTO 5300
c 100     CONTINUE
      ENDIF

5000  RETURN


! Error message, 
 5300 CALL CD_GET_VAR_INFO (dset, varid, vname, vtype,
     .                              nvdims, vdims, nvatts, cv, 
     .                              all_outflag, status)
      slen = TM_LENSTR( attname )
      wlen = TM_LENSTR( vname )
      risc_buff(1:slen) = attname
      risc_buff(slen+1:slen+14) = ' for variable '
      risc_buff(slen+15:slen+15+wlen) = vname
      CALL TM_ERRMSG (merr_attalready, status, 'CD_SET_ATT_FLAG',
     .           dset_num, no_stepfile, 
     .           risc_buff(1:slen+15+wlen),
     .           no_errstring, *5000)
      END
