	SUBROUTINE TM_FTOC_STRNG ( fstrng, cstrng, clen )

* convert a FORTRAN-style character string to a null-terminated Hollerith
* suitable for passing to a c routine
* normally the c string should be at least 1 byte longer than the FORTRAN
*    string to accomodate the null terminator
* note:  if the FORTRAN string contains terminating blanks they WILL be passed
*        to the c string.  Pass the FORTRAN string with its exact length if
*        this is not desired.
*   e.g. CALL TM_FTOC_STRNG ( fstrng(:flen), cstrng, clen )

* programmer - steve hankin
* NOAA/PMEL, Seattle, WA - Tropical Modeling and Analysis Program
* written for DECstation computer under Ultrix operating system
*
* rev 0.0 -   5/6/91
*  Sun Port 1/30/92 -kob-  added ifdef sun because uses BYTE where DECstation uses
*                          INTEGER*1
*calling arguments:
* fstrng - CHARACTER string (passed by descriptor)
* cstrng - BYTE (INTEREG*1) string (passed by pointer)
* clen   - length of passed Hollerith array

#ifdef unix
* calling argument declarations:
        INTEGER clen
        CHARACTER*(*) fstrng

#ifdef sun
        BYTE      cstrng(clen)
#else
	INTEGER*1 cstrng(clen)
#endif

* internal variable declarations:
	INTEGER	i, flen

* length of input string
        flen = LEN(fstrng)

* copy from CHARACTER to null-terminated Hollerith
        IF ( clen .GT. flen ) THEN
           DO 100 i = 1, flen
 100       cstrng(i) = ICHAR( fstrng(i:i) )
           cstrng(flen+1) = 0

        ELSE
* ... may be prematurely truncated ...
           DO 200 i = 1, clen-1
 200       cstrng(i) = ICHAR( fstrng(i:i) )
           cstrng(clen) = 0

        ENDIF

	RETURN
#else
        STOP 'TM_FTOC_STRNG is for Unix systems, only'
#endif
	END

