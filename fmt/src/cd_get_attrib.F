	LOGICAL FUNCTION CD_GET_ATTRIB ( cdfid, varid, attrib,
     .                                   do_warn, vname,
     .                                   string, slen )

* get an attribute string from a netCDF file if the attribute exists and
* is of the right data type
* the netCDF file is assumed to be open

* Programmer Steve Hankin
* NOAA/PMEL, Seattle, WA - Tropical Modeling and Analysis Program

* revision 0.0 - 2/92
*              - 8/92: fixed garbled error message text
* V420 - 11/95 *sh* - only issue warning if non-blanks are truncated
*		    - increase reading buffer size to 256

* argument definitions:
*	cdfid	 - netCDF id number for already opened CDF file
*       varid    - variable id in netCDF file
*       attrib   - attribute to search for
*	do_warn	 - logical flag to issue warning if invalid attribute
*                - warnings are never issued if attribute doesn't exits
*       vname    - name of variable owning attribute (used in error msgs)
*       string   - array into which to return result
*       slen     - maximum allowable return string

* argument declarations
      LOGICAL       do_warn
      INTEGER	    cdfid, varid, slen
      CHARACTER*(*) attrib, vname, string

#ifdef unix
      include 'tmap_format/netcdf.decl'
      include 'tmap_format/netcdf.inc'
      include 'tmap_format/xio.cmn_text'
#else
      INCLUDE 'TMAP_FORMAT:NETCDF.DECL'
      INCLUDE 'TMAP_FORMAT:NETCDF.INC'
      INCLUDE 'TMAP_FORMAT:XIO.CMN'
#endif

* internal variable declarations:
      LOGICAL too_long
      INTEGER TM_LENSTR1, lena, lenb, attype, attlen, cdfstat, i
      CHARACTER*256 buff

      INTEGER tt  ! nice short name
      EQUIVALENCE (tt,lunit_errors)

* initialize
       string(:slen) = ' '
       too_long = .FALSE.

* inquire about the attribute by name
! 11/95 note:  NCAINQ appears not to return a meaningful attlen -- always 256
      lena = TM_LENSTR1( attrib )
      CALL NCAINQ(cdfid, varid, attrib(:lena), attype, attlen, cdfstat )

      IF ( cdfstat .NE. NCNOERR ) THEN
         CD_GET_ATTRIB = .FALSE.
         RETURN  ! doesn't exist
      ENDIF

      IF ( attype.NE.NCCHAR .AND. attype.NE.NCBYTE ) THEN
         buff = 'attribute must be a string'
         CD_GET_ATTRIB = .FALSE.
         GOTO 5000
      ELSEIF ( attlen .GT. 256 ) THEN
         buff = 'attribute text too long'
         CD_GET_ATTRIB = .FALSE.
         GOTO 5000
      ELSE
         CALL NCAGTC(cdfid,varid,attrib,buff,256,cdfstat)
* ... f77 vs c bug:
* ... replace trailing nulls with blanks for FORTRAN strings
            DO 100 i = attlen, 1, -1
               IF ( ICHAR(buff(i:i)) .EQ. 0 ) THEN
               buff(i:i) = ' '
               ENDIF
 100     CONTINUE
         CD_GET_ATTRIB = .TRUE.
         string = buff
         IF ( TM_LENSTR1(buff(:attlen)) .GT. slen ) THEN
            too_long = .TRUE.
            buff = 'attribute truncated'	! warning msg
            GOTO 5000
	 ENDIF
      ENDIF


      RETURN

* warning exit
 5000 lenb = TM_LENSTR1( buff )
      buff = buff(:lenb)//': "'//attrib(:lena)//
     .             '" in netCDF variable: '//vname
      lenb = TM_LENSTR1( buff )
      IF ( do_warn ) CALL TM_NOTE( buff(:lenb), tt )
      IF ( do_warn.AND.too_long ) WRITE (tt,'(1X,A,I4)')
     .                              'Maximum characters: ',slen
      RETURN

      END
