	SUBROUTINE TM_ALTERNATE_NOTE ( string, lunit )

* send a warning message to an alternative source
* this routine can be replaced by a routine of choice as needed

* NOAA/PMEL, Seattle, WA - Tropical Modeling and Analysis Program
* written for VAX computer under VMS operating system

* *sh* 2/28/95

* calling argument declarations:
        INTEGER lunit
	CHARACTER*(*)	string

* internal variable declarations:
	INTEGER		TM_LENSTR1, llen

* internal parameter declarations:
        INTEGER      slen 
        PARAMETER  ( slen = 255 )

#ifdef sun
        BYTE      fhol(slen)	! c-type Hollerith string buffer
#else
        INTEGER*1 fhol(slen)	! c-type Hollerith string buffer
#endif

* true string length
	llen = TM_LENSTR1( string )

* convert to a null-terminated Hollerith
	CALL TM_FTOC_STRNG( string(:llen), fhol, slen )

* send the message to the alternative destination
	CALL FERRET_WARN_IN_WINDOW( fhol, 1 )     ! 1="append"

	RETURN
	END
