	LOGICAL FUNCTION ITSA_1LINEIF( cmnd, slen )

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
* determine whether the given command string is a one-line IF command of the
* form
*      IF 0_or_1 THEN command_which_include_ELSE [ENDIF optional]

* programmer - steve hankin
* NOAA/PMEL, Seattle, WA - Tropical Modeling and Analysis Program

* V693+  *sh* 12/14  (borrowing code from PROCESS_IF.F

        IMPLICIT NONE
* calling argument declarations
	CHARACTER*(*) cmnd
	INTEGER	slen

* local variable declarations
	INTEGER i, pos_then

        CHARACTER*1     tab
	PARAMETER     ( tab = CHAR(9))

* NOTE: this code **ASSUMES*
*  1) that the starting characters of the command string are "IF "
*  2) that there is no bang-comment string
*  3) that the string is already up-cased


* the test is simply that the command contains "THEN" and there is further
* text found after it

	pos_then = INDEX(cmnd(:slen), 'THEN')
	IF ( pos_then .GT. 0 ) THEN
	   DO i = pos_then+4, slen
              IF (cmnd(i:i).NE. ' ' .AND. cmnd(i:i).NE. tab) GOTO 100
	   ENDDO
* ... no characters found after THEN -- this is not a one-liner
	ENDIF

	ITSA_1LINEIF = .FALSE.
	RETURN

* we found a "THEN" and we also found more text after it
 100	ITSA_1LINEIF = .TRUE.
	RETURN

	END
