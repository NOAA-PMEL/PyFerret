	SUBROUTINE TM_FTOC_STRNG ( fstrng, cstrng, clen )

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
* 11/2022 *acm* Code cleanup: Modernize DO-loop syntax. Clean up obsolete ifdefs
*calling arguments:
* fstrng - CHARACTER string (passed by descriptor)
* cstrng - BYTE (INTEREG*1) string (passed by pointer)
* clen   - length of passed Hollerith array

* calling argument declarations:
        INTEGER clen
        CHARACTER*(*) fstrng
	INTEGER*1 cstrng(clen)

* internal variable declarations:
	INTEGER	i, flen

* length of input string
        flen = LEN(fstrng)

* copy from CHARACTER to null-terminated Hollerith
        IF ( clen .GT. flen ) THEN
           DO i = 1, flen
              cstrng(i) = ICHAR( fstrng(i:i) )
           ENDDO
           cstrng(flen+1) = 0

        ELSE
* ... may be prematurely truncated ...
           DO i = 1, clen-1
              cstrng(i) = ICHAR( fstrng(i:i) )
 	   ENDDO
           cstrng(clen) = 0

        ENDIF

	RETURN

	END

