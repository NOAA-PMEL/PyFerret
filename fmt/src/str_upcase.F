      INTEGER FUNCTION STR_UPCASE( str_out, str_in )

* FORTRAN-coded version of VAX runtime library
* convert string to uppercase

* programmer - steve hankin
* NOAA/PMEL, Seattle, WA - Tropical Modeling and Analysis Program
* written for VAX computer under VMS operating system
*
* revision 0.0 -  5/ 6/91
* 10/96 *kob* - Linux port 
*		- had to declare hex constants as implicit
*		  integers in DATA statements.  Added a 
*		  FORTRAN_90 ifdef to accomplish this. 
*		- Needs to use IAND rather than AND
* calling argument declarations:
      CHARACTER*(*) str_out, str_in

* local variable declarations:
      INTEGER   slenout, slenin, i

* local parameter declarations:
      INTEGER     SS__NORMAL
      PARAMETER ( SS__NORMAL = 1 )

#ifdef FORTRAN_90
      INTEGER HEXDF
      DATA HEXDF/Z'DF'/
#endif

* initialize
      slenout = LEN(str_out)
      slenin  = LEN(str_in)

      DO 100 i = 1, MIN(slenout, slenin)

         IF ( str_in(i:i) .LT. 'a' .OR. str_in(i:i) .GT. 'z' ) THEN

            str_out(i:i) = str_in(i:i)

         ELSE

#if defined unix  && ! defined NEED_IAND
            str_out(i:i) = CHAR( AND( 'DF'X, ICHAR(str_in(i:i)) ) )
#else
# ifdef FORTRAN_90
            str_out(i:i) = CHAR( IAND( HEXDF, ICHAR(str_in(i:i)) ) )
# else
* VMS needs IAND
            str_out(i:i) = CHAR( IAND( 'DF'X, ICHAR(str_in(i:i)) ) )
# endif
#endif
         ENDIF

 100  CONTINUE

* pad the rest of the output string with blanks if extra long
* (IF test is redundant ... but safe)
      IF ( slenout .GT. slenin ) THEN
         DO 200 i = slenin+1, slenout
 200     str_out(i:i) = ' '
      ENDIF

* successful return always
      STR_UPCASE = SS__NORMAL
      RETURN
      END
