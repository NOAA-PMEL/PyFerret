      INTEGER FUNCTION STR_CASE_BLIND_COMPARE( str1, str2 )

* FORTRAN-coded version of VAX runtime library
* compare two strings disregarding case differences

* programmer - steve hankin
* NOAA/PMEL, Seattle, WA - Tropical Modeling and Analysis Program
*
* revision 0.0 -  1/20/91
* 10/96 *kob* - Linux port 
*		- had to declare hex constants as implicit
*		  integers in DATA statements.  Added a 
*		  FORTRAN_90 ifdef to accomplish this. 
*		- Needs to use IAND rather than AND
* calling argument declarations:
      CHARACTER*(*) str1, str2

* local variable declarations:
      INTEGER   slen, i
      CHARACTER*1  c1, c2

* local parameter declarations:
      INTEGER     str1_less,  str_equal, str1_more
      PARAMETER ( str1_less = -1,
     .            str_equal =  0,
     .            str1_more =  1 )
#ifdef FORTRAN_90
      INTEGER HEXDF
      DATA HEXDF/Z'DF'/
#endif
      slen = MIN( LEN(str2), LEN(str1) )

* to begin: compare through the last character of the shorter string
      DO 100 i = 1, slen

         c1 = str1(i:i)
         c2 = str2(i:i)

* immediate check for equal to spped things up
         IF ( c1 .EQ. c2 ) GOTO 100

* up-case the lower case characters
#if defined unix  && ! defined NEED_IAND

         IF ( c1 .GE. 'a' .AND. c1 .LE. 'z' )
     .                  c1 = CHAR( AND( 'DF'X, ICHAR(c1) ) )

         IF ( c2 .GE. 'a' .AND. c2 .LE. 'z' )
     .                  c2 = CHAR( AND( 'DF'X, ICHAR(c2) ) )
#else
# ifdef FORTRAN_90
         IF ( c1 .GE. 'a' .AND. c1 .LE. 'z' )
     .                  c1 = CHAR( IAND( HEXDF, ICHAR(c1) ) )

         IF ( c2 .GE. 'a' .AND. c2 .LE. 'z' )
     .                  c2 = CHAR( IAND( HEXDF, ICHAR(c2) ) )
# else
* VMS requires "IAND" ...
         IF ( c1 .GE. 'a' .AND. c1 .LE. 'z' )
     .                  c1 = CHAR( IAND( 'DF'X, ICHAR(c1) ) )

         IF ( c2 .GE. 'a' .AND. c2 .LE. 'z' )
     .                  c2 = CHAR( IAND( 'DF'X, ICHAR(c2) ) )
# endif
#endif

* try comparison again
         IF ( c1 .EQ. c2 ) GOTO 100

* nope - they're different !
         IF ( c1 .LT. c2 ) THEN
            STR_CASE_BLIND_COMPARE = str1_less
         ELSE
            STR_CASE_BLIND_COMPARE = str1_more
         ENDIF
         RETURN

 100  CONTINUE

* they're equal through the length of the shorter string
* any non-blanks that remain in either string represent differences
      IF ( LEN(str1) .GT. slen ) THEN
         DO 200 i = slen+1, LEN(str1)
            IF (str1(i:i) .NE. ' ' ) THEN 
               STR_CASE_BLIND_COMPARE = str1_more
               RETURN
            ENDIF
 200     CONTINUE
      ELSEIF ( LEN(str2) .GT. slen ) THEN
         DO 210 i = slen+1, LEN(str2)
            IF (str2(i:i) .NE. ' ' ) THEN
               STR_CASE_BLIND_COMPARE = str1_less
               RETURN
            ENDIF
 210     CONTINUE
      ENDIF

* text is equal even though the string lengths may be different
      STR_CASE_BLIND_COMPARE = str_equal

      RETURN
      END
