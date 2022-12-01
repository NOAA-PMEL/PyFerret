	SUBROUTINE TM_CTOF_STRNG ( cstrng, fstrng, clen )

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
* convert a null-terminated Hollerith (suitable for passing to a c routine) to
* a FORTRAN-style character string 
* normally the c string should be at least 1 byte longer than the FORTRAN
*    string to accomodate the null terminator

* programmer - steve hankin
* NOAA/PMEL, Seattle, WA - Tropical Modeling and Analysis Program
* written for DECstation computer under Ultrix operating system
*
* rev 0.0 -   5/6/91
*  Sun Port 1/30/92 -kob-  added ifdef sun because uses BYTE where DECstation uses
*                          INTEGER*1
*
* IBM PORT 14apr94 *kob* : IBM compiler fussy about argument types
*
* Linux port 10/96 *kob*: for some reason, cstrng was declared as a 
*			  type byte even if the machine was not a 
*			  sun...this is wrong and I changed it back
* 11/2022 *acm* Code cleanup: Modernize DO-loop syntax. 
*		Remove obsolete ifdefs.

* calling arguments:
* cstrng - BYTE (INTEREG*1) string (passed by pointer)

* fstrng - CHARACTER string (passed by descriptor)
* clen   - length of passed Hollerith array

* calling argument declarations:
        INTEGER clen
        CHARACTER*(*) fstrng
	INTEGER*1 cstrng(clen)			!*kob* 10/96

* internal variable declarations:
	INTEGER	i, flen

* initialize
        flen = LEN(fstrng)
        fstrng = ' '

* copy from null-terminated Hollerith to CHARACTER
        DO i = 1, MIN(clen, flen)
           IF ( cstrng(i) .EQ. 0 ) RETURN
           fstrng(i:i) = CHAR(cstrng(i))
	ENDDO

	RETURN
	END

