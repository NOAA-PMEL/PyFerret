      SUBROUTINE CD_GET_VAR_ATT_ID (dset, varid, attname, attid, status)

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
* given dset, variable id, and attribute name return attribute id
* Test the attribute name in a case-insensitve manner.

* Programmer Ansley Manke
* NOAA/PMEL, Seattle, WA - TMAP

* V600 6/05 *acm* Attribute control.
*      6/06 *acm* missing declarations of ncf functions


* include files
#include "gt_lib.parm"
      INCLUDE "tmap_errors.parm"

* Argument declarations

      CHARACTER*(*) attname
      INTEGER dset, varid, attid, status

* Internal declarations

        include 'netcdf.decl'
#include "netcdf.inc"

      CHARACTER*512 abuff
      LOGICAL coordvar
      INTEGER TM_LENSTR1, NCF_GET_VAR_ID, NCF_GET_VAR_ATTR_ID, 
     .        NCF_GET_VAR_ATTR_ID_CASE, STR_SAME, 
     .        slen, flen, dset_num
      PARAMETER (flen=512)      ! size of variable name string

#ifdef sun
      BYTE      fhol(flen)      ! c-type Hollerith string buffer
#else
      INTEGER*1 fhol(flen)      ! c-type Hollerith string buffer
#endif 

      attid = 0
      status = atom_not_found

      dset_num = dset
      IF (dset_num .LT. -1) dset_num = -1

      abuff = ""
      abuff = attname

      slen = TM_LENSTR1( abuff )

* If the name is a pseudo-name for describing attributes then return attid=0.
* If a file contains one of these names as an attribute, it can be accessed
* by putting it in quotes.

      IF (STR_SAME(abuff(:slen), 'varnames') .EQ. 0 ) THEN
         RETURN
      ELSE IF (STR_SAME(abuff(:slen), 'nvars') .EQ. 0 ) THEN
         RETURN
      ELSE IF (STR_SAME(abuff(:slen), 'attnames') .EQ. 0 ) THEN 
         RETURN
      ELSE IF (STR_SAME(abuff(:slen), 'nattrs') .EQ. 0 ) THEN
         RETURN
      ELSE IF (STR_SAME(abuff(:slen), 'dimnames') .EQ. 0 ) THEN
         RETURN
      ELSE IF (STR_SAME(abuff(:slen), 'ndims') .EQ. 0 ) THEN
         RETURN
      ELSE IF (STR_SAME(abuff(:slen), 'nctype') .EQ. 0 ) THEN
         RETURN
      ENDIF

* If the name is in quotes, remove them and do a case-sensive search
* for the name.

      slen = TM_LENSTR1( abuff )
      IF (abuff(1:1) .EQ. "'" .AND. abuff(slen:slen) .EQ. "'" ) THEN
         abuff = abuff(2:slen-1)
         abuff(slen-1:slen) = ""
         slen = slen - 2
         CALL TM_FTOC_STRNG( abuff(1:slen), fhol, flen )
         status = NCF_GET_VAR_ATTR_ID_CASE(dset_num, varid, fhol, attid)
      ELSE 
         CALL TM_FTOC_STRNG( abuff(1:slen), fhol, flen )
         status = NCF_GET_VAR_ATTR_ID(dset_num, varid, fhol, attid)
      ENDIF
      
      dset = dset_num  ! if changed above, then dset=-1 is correct

 5000 RETURN
      END
