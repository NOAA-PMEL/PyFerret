	LOGICAL FUNCTION CD_GET_ATTVAL ( cdfid, varid, attrib,
     .                                   do_warn, vname,
     .                                   val, vlen, vback )

* get numerical attributes from a netCDF file if the attribute exists
* convert all attribute values to floating point
* the netCDF file is assumed to be open
* Note: this routine is "quick and dirty" - long arrays of non-float type
*       or high double precision exponents may blow it up

* Programmer Steve Hankin
* NOAA/PMEL, Seattle, WA - Tropical Modeling and Analysis Program

* revision 0.0 - 2/92
*
* 11/96 *kob*  - Linux Port
*	       - had to add an real*4 buffer to the buff equivalence.  
*		 Linux f77 complained aobut inconsistent datatypes
*
* 4/98  *kob*  - fix bug introduced by above.  Linux f90 compiler will not allow
*                r8 buff to be passed to NCAGT, complaining of inconsistent 
*                datatypes.  SO, r4buff is passed to NCAGT, but really only as the
*                address pointer.  then, when the values are actually used, r8buff
*                is the buffer from which they are read.  this allows dp attribute
*                values to be used correctly.

* argument definitions:
*	cdfid	 - netCDF id number for already opened CDF file
*       varid    - variable id in netCDF file
*       attrib   - attribute to search for
*	do_warn	 - logical flag to issue warning if invalid attribute
*       vname    - name of variable owning attribute (used in error msgs)
*       val      - array into which to return result
*       vlen     - maximum allowable return array len
*       vback    - number of elements in return array

* argument declarations
      LOGICAL       do_warn
      INTEGER	    cdfid, varid, vlen, vback
      CHARACTER*(*) attrib, vname
      REAL val(vlen)

#ifdef unix
      include 'tmap_format/netcdf.decl'
      include 'tmap_format/netcdf.inc'
#  ifdef sun
      BYTE i1buff(132)
#  else
      INTEGER*1 i1buff(132)
#  endif
#else
      BYTE i1buff(132)
      INCLUDE 'TMAP_FORMAT:NETCDF.DECL'
      INCLUDE 'TMAP_FORMAT:NETCDF.INC'
#endif

* internal variable declarations:
      INTEGER TM_LENSTR1, lena, attype, cdfstat, i
      CHARACTER*132 buff

* equivalent buffer arrays for various data types
      INTEGER*2 i2buff(64)
      INTEGER*4 i4buff(32)
      REAL*4 	r4buff(32)		!kob 11/96
      REAL*8    r8buff(16)
      EQUIVALENCE (buff,i1buff),
     .            (buff,i2buff),
     .            (buff,i4buff),
     .            (buff,r4buff),	!kob 11/96
     .            (buff,r8buff)

* inquire about the attribute by name
      lena = TM_LENSTR1( attrib )
      CALL NCAINQ(cdfid, varid, attrib(:lena), attype, vback, cdfstat )

      IF ( cdfstat .EQ. NCNOERR ) THEN
         IF ( vback .GT. vlen ) THEN
            buff = 'too many values in attribute "'//attrib(:lena)//
     .             '" in netCDF file variable: '//vname
            lena = TM_LENSTR1( buff )
            IF ( do_warn ) CALL TM_NOTE( buff(:lena), 6 )
            CD_GET_ATTVAL = .FALSE.
         ELSEIF (attype.EQ.NCFLOAT) THEN
            CALL NCAGT(cdfid,varid,attrib,val,cdfstat)
            CD_GET_ATTVAL = .TRUE.
         ELSE
! *kob* 4/98 - see comment above
!            CALL NCAGT(cdfid,varid,attrib,r8buff,cdfstat)
            CALL NCAGT(cdfid,varid,attrib,r4buff,cdfstat)   !kob 4/98
            CD_GET_ATTVAL = .TRUE.
* convert data types
            IF (attype.EQ.NCBYTE .OR. attype.EQ.NCCHAR ) THEN
               DO 100 i = 1, vback
 100           val(i) = i1buff(i)
            ELSEIF (attype.EQ.NCSHORT) THEN
               DO 200 i = 1, vback
 200           val(i) = i2buff(i)
            ELSEIF (attype.EQ.NCLONG ) THEN
               DO 300 i = 1, vback
 300           val(i) = i4buff(i)
            ELSE
               DO 400 i = 1, vback
 400           val(i) = r8buff(i)  
! 400           val(i) = r4buff(i)   ! kob 4/98
            ENDIF
         ENDIF
      ELSE
         CD_GET_ATTVAL = .FALSE.
         vback = 0
      ENDIF

      RETURN
      END
