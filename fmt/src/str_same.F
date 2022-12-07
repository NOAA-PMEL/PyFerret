      INTEGER FUNCTION STR_SAME( str1, str2 )

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
* compare two strings disregarding spaces and case differences

* programmer - ANsley Manke (from str_case_blind_compare) 
*
* revision 0.0 -  4/02/03

* calling argument declarations:
      CHARACTER*(*) str1, str2

* local variable declarations:
      INTEGER      s1, s2, e1, e2
      CHARACTER*1  c1, c2

* local parameter declarations:
      INTEGER     STR_CASE_BLIND_COMPARE


* local parameter declarations:
        CHARACTER*1     tab
	PARAMETER     ( tab = CHAR(9))

      e1 = LEN(str1)
      e2 = LEN(str2)

* skip leading blanks

      DO 100 s1 = 1, LEN(str1)
         c1 = str1(s1:s1)
         IF ( c1 .NE. ' ' .AND. c1 .NE. tab ) GOTO 200
 100  CONTINUE
 200  CONTINUE

      DO 300 s2 = 1, LEN(str2)
         c2 = str2(s2:s2)
         IF ( c2 .NE. ' ' .AND. c2 .NE. tab ) GOTO 400
  300 CONTINUE
  400 CONTINUE

      STR_SAME = STR_CASE_BLIND_COMPARE (str1(s1:e1), str2(s2:e2) )

      RETURN
      END
