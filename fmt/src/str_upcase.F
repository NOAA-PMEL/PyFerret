      INTEGER FUNCTION STR_UPCASE( str_out, str_in )

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
