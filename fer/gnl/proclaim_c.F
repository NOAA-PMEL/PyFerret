	SUBROUTINE PROCLAIM_C( lun, leader ) 

* pass arguments from C language to the "PROCLAIM" routine
* make a movie frame of the current output graphics

* programmer - steve hankin
* NOAA/PMEL, Seattle, WA - Tropical Modeling and Analysis Program
*
* revision 0.0 - 8/94


* calling argument declarations
	INTEGER lun
	INTEGER	slen
        PARAMETER  ( slen = 32 )
#ifdef sun
        BYTE      leader(slen) ! c-type Hollerith string buffer
#else
        INTEGER*1 leader(slen) ! c-type Hollerith string buffer
#endif

* internal variable declarations:
	INTEGER	TM_LENSTR1, flen
	CHARACTER*32 fstrng

	CALL TM_CTOF_STRNG( leader, fstrng, slen )
	flen = TM_LENSTR1(fstrng)
	CALL PROCLAIM( lun, fstrng(:flen) )
	RETURN
	
	END

