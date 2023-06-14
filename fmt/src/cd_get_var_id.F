      SUBROUTINE CD_GET_VAR_ID (dset, vname, varid, status)

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
* given dset and variable name return variable id

* Programmer Ansley Manke
* NOAA/PMEL, Seattle, WA - TMAP

* V600 6/05 *acm* Attribute control.
*      6/06 *acm* missing declarations of ncf functions
* V602  2/07 *acm* Fix bug 1492, changing attributes of coordinate variables 
*           *acm* Get rid of call to isit_coord_var, which could get us in 
*                 an endless loop (isit_coord_var calls cd_get_var_id).
* V611  5/08 *acm* Fix bug 1578, dataset specifier on the variable var[d=1].attr
* V6.74 3/12 *acm* cleanup ifdefs and unnecessary include files
* V6.94 3/15 *acm* Fix ticket 2249: attribute handling for user-defined variables
* V698 *sh* 1/16 -- added status check/warning, since not checked by callers
* V699 *sh* 6/16 - bug fix:  this routine falsely identified Ferret child
*                  variables, which have names of the form "(Cnnn,Vmmm)" as
*                  coordinate variables.  It then returned a status indicating
*                  failure that was regarded as acceptable, presumably because
*                  a user may mis-type a coordinate variable name
*                  The result is that child variables did not get deleted.
* V7   *acm* 6/16 -The above fix needs to first check for string length.
* V7   *acm* 3/20 -Issue 1969: the case-sensitive match for names in quotes
*                  needs to be done on the incoming name not its upcased version.

* include files
#include "gt_lib.parm"
      INCLUDE "tmap_errors.parm"

* Argument declarations

      CHARACTER*(*) vname
      INTEGER dset, varid, status

* Internal declarations

      INCLUDE 'netcdf.inc'

      CHARACTER*512 buff
      LOGICAL coordvar
      INTEGER TM_LENSTR1, NCF_GET_VAR_ID, NCF_GET_VAR_ID_CASE,
     .        NCF_GET_VAR_AXFLAG, FIND_DSET_NUMBER, STR_UPCASE,
     .        slen, flen, dir, dset_num, bkt1, bkt2, istrt, iend, iset
      PARAMETER (flen=512)      ! size of variable name string
      INTEGER*1 fhol(flen)      ! c-type Hollerith string buffer

      varid = 0
      status = atom_not_found

      dset_num = dset
      IF (dset_num .LT. -2) dset_num = -2

      istrt = STR_UPCASE(buff,vname)

* 6/16 *sh* **HACK!!!**
* ==> Detect Ferret child-variable name of the form "(C011,V007)"
* so they are not mis-identified as coordinate variable names
* (IMHO a low level routine such as this one should not be doing parsing)
      slen = TM_LENSTR1(vname)
      IF (slen .GE. 11) THEN
         IF  ( vname(1:1)   .EQ. '('
     .    .AND.vname(2:2)   .EQ. 'C'
     .    .AND.vname(6:6)   .EQ. ','
     .    .AND.vname(7:7)   .EQ. 'V'
     .    .AND.vname(11:11) .EQ. ')'  ) GOTO 500
      ENDIF

* Deal with coordinate variable names inside parenthese

      IF (vname(1:1) .EQ. '(' ) THEN
         slen = INDEX(buff,')') - 1 
         IF (slen .LE. 0) slen = TM_LENSTR1(buff)
         buff = vname(2:slen)
         buff(slen:slen) = ""
         slen = slen - 1
       
         CALL TM_FTOC_STRNG( buff(1:slen), fhol, flen )
         status = NCF_GET_VAR_ID(dset_num, varid, fhol)
         IF (status.NE.merr_ok .AND.
     .       INDEX(vname,'[') .GT. 0) GOTO 4000
         status = NCF_GET_VAR_AXFLAG (dset, varid, coordvar, dir)
         IF (.NOT. coordvar) THEN
	    status = atom_not_found 
	    GOTO 5000
	 ENDIF

* If theres also a dataset spec (xax)[d=1] on the orig variable name, 
* add that
 4000    CONTINUE
         IF ( INDEX(vname,'[') .GT. 0 ) THEN
            bkt1 = INDEX(vname, '[')
            bkt2 = INDEX(vname, ']')
            slen = TM_LENSTR1(buff)
            buff = buff(1:slen)//vname(bkt1:bkt2)
         ENDIF

      ENDIF 

* If there is a dataset spec, see if the variable is in that dataset.
* Look only at the d=n or d=dset_name. 
* Otherwise ignore what is in the brackets; this allows us to get
* attributes for LAS variables with region info.

      IF ( INDEX(buff,'[').GT.0) THEN
         
         istrt = STR_UPCASE(buff,buff)
	 istrt = INDEX(buff,'D=')
	 IF (istrt .GT. 0) THEN
	    istrt = INDEX(buff,'D=') + 2
	    iend = INDEX(buff(istrt:),',') - 1
	    IF (iend .LE. 0) THEN
	       iend = INDEX(buff,']') - 1
	    ELSE
	       iend = istrt+iend-1
	    ENDIF
	    iset = FIND_DSET_NUMBER(buff(istrt:iend))
	    IF (iset .NE. unspecified_int4) dset_num = iset
	 ENDIF
	 
	 istrt = INDEX(buff,'[') - 1
	 buff = buff(:istrt)
	 slen = TM_LENSTR1(buff)
      ENDIF

* If the name is in quotes, remove them, from the un-upcased version, and do a 
* case-sensitive search for the name.

 500  slen = TM_LENSTR1( buff )
      IF (buff(1:1) .EQ. "'" .AND. buff(slen:slen) .EQ. "'" ) THEN
         CALL TM_FTOC_STRNG( vname(2:slen-1), fhol, flen )
         status = NCF_GET_VAR_ID_CASE(dset_num, varid, fhol)
      ELSE
         CALL TM_FTOC_STRNG( buff(1:slen), fhol, flen )
         status = NCF_GET_VAR_ID(dset_num, varid, fhol) 
      ENDIF
      IF (status .NE. merr_ok) GOTO 5000
      dset = dset_num

* If the name is just a dot, it is the fake global-attributes variable

      IF (slen.EQ.1 .AND. buff(1:1) .EQ. ".") THEN
         varid = 0
	 status = merr_ok
      ENDIF
      RETURN

* error exit
 5000 IF (status .NE. atom_not_found) THEN
	slen = TM_LENSTR1(buff)
	CALL WARN('Error status from CD_GET_VAR_ID: '//buff(1:slen))
      ENDIF
      RETURN
      END
