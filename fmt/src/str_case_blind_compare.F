      INTEGER FUNCTION STR_CASE_BLIND_COMPARE( str1, str2 )

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
