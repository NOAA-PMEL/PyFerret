	CHARACTER*1 FUNCTION UPPER_CASE ( input_char )

* convert character string to upper case

* programmer - steve hankin
* NOAA/PMEL, Seattle, WA - Tropical Modeling and Analysis Program
* written for VAX computer under VMS operating system
*
* revision 0.0 - 4/4/86
* Unix/RISC port - 2/13/91 - .AND. always produces a logical result

* 5/95 *sh* - added ifdef for HP port
* 12/96 *kob* - Linux port 
*               - had to declare hex constants as implicit
*                 integers in DATA statements.  Added a 
*                 FORTRAN_90 ifdef to accomplish this. 
*               - Needs to use IAND rather than AND


	CHARACTER*1 input_char

* kob 12/96
#ifdef FORTRAN_90
      INTEGER HEXDF
      DATA HEXDF/Z'DF'/
#endif

	IF ( input_char .LT. 'a' .OR. input_char .GT. 'z' ) THEN
	   upper_case = input_char
	ELSE
* modify if logic for linux - also vms needs iand - like that
* matters anymore - just to keep it clean *kob* 12/96

#if defined unix  && !defined NEED_IAND
           upper_case = CHAR( AND('DF'X , ICHAR( input_char ) ) )
#else
#  ifdef FORTRAN_90 			
	   upper_case = CHAR( IAND(HEXDF, ICHAR( input_char ) ) )
#  else
	   upper_case = CHAR( IAND('DF'X , ICHAR( input_char ) ) )
#  endif
#endif
	ENDIF

	RETURN
	END
