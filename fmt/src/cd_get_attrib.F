	LOGICAL FUNCTION CD_GET_ATTRIB ( cdfid, varid, attrib,
     .                                   do_warn, vname,
     .                                   string, attlen, slen )

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
* get an attribute string from a netCDF file if the attribute exists and
* is of the right data type
* the netCDF file is assumed to be open

* Programmer Steve Hankin
* NOAA/PMEL, Seattle, WA - Tropical Modeling and Analysis Program

* revision 0.0 - 2/92
*              - 8/92: fixed garbled error message text
* V420 - 11/95 *sh* - only issue warning if non-blanks are truncated
*		    - increase reading buffer size to 256
* V581 -  5/05 *acm - increase reading buffer size to 1024.  Newer netCDF 
*                     libraries have apparently lifted the restriction.
*                     This lets us increase the size of e.g. the title attribute.
* V602 *acm* make sure lena=length of attr name is defined before writing warning
* V62  6/08 *acm* Increase reading buffer size to 2048 and check the status flag when
*            reading the attribute values.
* V65  *acm* 1/10  Change all netcdf2-style calls and parameter names to 
*                  netcdf3-style calls, which are also consistent with netcdf4.

* argument definitions:
*	cdfid	 - netCDF id number for already opened CDF file
*       varid    - variable id in netCDF file
*       attrib   - attribute to search for
*	do_warn	 - logical flag to issue warning if invalid attribute
*                - warnings are never issued if attribute doesn't exits
*       vname    - name of variable owning attribute (used in error msgs)
*       string   - array into which to return result
*       attlen   - returned length of return string
*       slen     - maximum allowable return string

* argument declarations
      LOGICAL       do_warn
      INTEGER	    cdfid, varid, attlen, slen
      CHARACTER*(*) attrib, vname, string

      include 'netcdf.decl'
#include "netcdf.inc"
      include 'xio.cmn_text'

* internal variable declarations:
      LOGICAL too_long
      INTEGER TM_LENSTR1, lena, lenb, attype, cdfstat, i
      CHARACTER*2048 buff

      INTEGER tt  ! nice short name
      EQUIVALENCE (tt,lunit_errors)

* initialize
       string(:slen) = ' '
       too_long = .FALSE.

* inquire about the attribute by name
! 11/95 note:  NCAINQ appears not to return a meaningful attlen -- always 256
      lena = TM_LENSTR1( attrib )
      cdfstat = NF_INQ_ATT (cdfid, varid, attrib(:lena), attype, attlen )

      IF ( cdfstat .NE. NF_NOERR ) THEN
         attlen = 0
         CD_GET_ATTRIB = .FALSE.
         RETURN  ! doesn't exist
      ENDIF

      IF ( attype.NE.NF_CHAR .AND. attype.NE.NF_BYTE ) THEN
         buff = 'attribute must be a string'
         CD_GET_ATTRIB = .FALSE.
         GOTO 5000
      ELSE
         cdfstat = NF_GET_ATT_TEXT (cdfid, varid, attrib, buff)
         IF (cdfstat .NE. NF_NOERR ) THEN
            CD_GET_ATTRIB = .FALSE.
            GOTO 5000
         ELSE
* ... f77 vs c bug:
* ... replace trailing nulls with blanks for FORTRAN strings
            DO 100 i = attlen, 1, -1
               IF ( ICHAR(buff(i:i)) .EQ. 0 ) THEN
               buff(i:i) = ' '
               ENDIF
 100        CONTINUE
            CD_GET_ATTRIB = .TRUE.
            string = buff
            IF ( TM_LENSTR1(buff(:attlen)) .GT. slen ) THEN
               too_long = .TRUE.
               buff = 'attribute truncated'	! warning msg
               GOTO 5000
            ENDIF

	 ENDIF
      ENDIF

      RETURN

* warning exits
 5000 lenb = TM_LENSTR1( buff )
      lena = TM_LENSTR1( attrib )
      buff = buff(:lenb)//': "'//attrib(:lena)//
     .             '" in netCDF variable: '//vname
      lenb = TM_LENSTR1( buff )
      IF ( do_warn ) CALL TM_NOTE( buff(:lenb), tt )
      IF ( do_warn.AND.too_long ) WRITE (tt,'(1X,A,I4)')
     .                              'Maximum characters: ',slen
      RETURN

      END
