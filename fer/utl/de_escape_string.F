	SUBROUTINE DE_ESCAPE_STRING ( string, slen )

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
* remove one set of backslash-escapes from a string thereby "exposing" the
* chat=racter that follows each backslash

* programmer - steve hankin
* NOAA/PMEL, Seattle, WA - Tropical Modeling and Analysis Program
* written for VAX computer under VMS operating system
*
* revision 0.0 - 4/4/86
* Unix/RISC port - 2/13/91 - .AND. always produces a logical result

* V510: 3/00 *sh*
* 11/2022 *acm* Code cleanup: Modernize DO-loop syntax. 

        IMPLICIT NONE
* calling argument declarations
	CHARACTER string*(*)

* internal variable declarations
	INTEGER	TM_LENSTR1, slen, iin, iout

! initialize
	slen = TM_LENSTR1(string)
	iin = 1
	iout = iin

! a DO-WHILE loop
 10	IF ( iin .GT. slen ) GOTO 50

!    ... do not copy a slash -- but always copy the character that follows it
#ifdef NO_DOUBLE_ESCAPE_SLASH
        IF (string(iin:iin) .EQ. '\' ) iin = iin + 1
#else
        IF (string(iin:iin) .EQ. '\\') iin = iin + 1
#endif
	
	IF ( iin .LE. slen ) THEN
	   string(iout:iout) = string(iin:iin)
	   iin = iin + 1
	   iout = iout + 1
	   GOTO 10
	ENDIF

!   blank out any remaining characters
 50	DO iin = iout, slen
	   string(iin:iin) = " " 
 	ENDDO
	slen = MAX(iout-1, 1)	! new string length
	RETURN

	END
