	LOGICAL FUNCTION NC_GET_ATTRIB_DP ( dset, varid, attrib,
     .                                   do_warn, vname, attlen, 
     .                                   attoutflag, dvals )

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
* get double-precision attribute values from a netCDF data structure if
* the attribute exists
* 
* Based on CD_GET_ATTRIB
* V600 -  5/05 *acm*  Attribute handling upgrade. Rework to read information 
*                     from data strucure rather than the netCDF file.
*                     Replace NCAINQ with CD_GET_VAR_ATT_INFO
*                     Replace NCAGTC with CD_GET_VAR_ATTR
*                     This routine now returns both character and real results,
*                     so it replaces CD_GET_ATTVAL.
*         6/06 *acm*  replace istat with status
* V602  2/07 *acm* Fix bug 1492, changing attributes of coordinate variables 
* V602 *acm* make sure lena=length of attr name is defined before writing warning
* V63  *acm* 10/09 Changes for gfortran build
* V65  *acm* 1/10  Make sure parameters values are consistent with 
*                              whats in netcdf.inc from netcdf4.
* V6.74 3/12 *acm* cleanup ifdefs and unnecessary include files

* argument definitions:
*	dset	 - dataset number
*       varid    - variable id 
*       attrib   - attribute to search for
*	do_warn	 - logical flag to issue warning if invalid attribute
*                - warnings are never issued if attribute doesn't exit
*       vname    - name of variable owning attribute (used in error msgs)
*       attlen   - length of output (string length or number of values)
*       attoutflag- 1 if attribute to be written to output files, 0 if not
*       dvals    - array in which to return result if numerical type

* argument declarations
      LOGICAL       do_warn
      INTEGER	    dset, varid, attlen, attoutflag
      CHARACTER*(*) attrib, vname
      real*8        dvals(*)

#include "gt_lib.parm"
      include 'netcdf.inc'
      include 'xio.cmn_text'
      include 'tmap_errors.parm'
      include 'xrisc_buff.cmn'

* internal variable declarations:
      LOGICAL too_long
      INTEGER TM_LENSTR1, NCF_GET_VAR_ATTR,
     .        lena, lenb, attype, status, i, dset_num, attid
      CHARACTER*128 aname

      INTEGER tt  ! nice short name
      EQUIVALENCE (tt,lunit_errors)

      INTEGER slen, flen
      PARAMETER (flen=size_rbuff)        ! size of attribute string
      INTEGER*1 fhol(flen), ghol(flen)     ! c-type Hollerith string buffer

       NC_GET_ATTRIB_DP = .FALSE.
       IF (dset .EQ. unspecified_int4) RETURN
       IF (varid .LT. 0) RETURN

* initialize
       too_long = .FALSE.

* inquire about the attribute by name, using its id number

      CALL CD_GET_VAR_ATT_ID (dset, varid, attrib, attid, status)
      IF (attid .GT. 0) CALL CD_GET_VAR_ATT_INFO (dset, varid, attid,
     .                   aname, attype, attlen, attoutflag, status )

      IF ( status .NE. merr_ok ) THEN
         NC_GET_ATTRIB_DP = .FALSE.
         RETURN  ! doesn't exist
      ENDIF

      lena = TM_LENSTR1( aname )
      CALL TM_FTOC_STRNG (aname(:lena), ghol, flen)

* Get attribute length and value

      dset_num = dset
      IF (dset_num .LT. -2) dset_num = -2
      status = NCF_GET_VAR_ATTR(dset_num, varid, ghol, fhol, attlen, dvals)

      IF ( status .NE. merr_ok ) THEN
         NC_GET_ATTRIB_DP = .FALSE.
         attlen = 0
         attoutflag = 0
         RETURN  
      ENDIF

      NC_GET_ATTRIB_DP = .TRUE.

      RETURN

      END
