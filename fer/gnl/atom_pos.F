	INTEGER FUNCTION ATOM_POS ( string, atom )

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
* find the position of the atom within the string
* if the atom is the last character in the string this represents an error

* programmer - steve hankin
* NOAA/PMEL, Seattle, WA - Tropical Modeling and Analysis Program
* written for VAX computer under VMS operating system
*
* revision 0.0 - 10/23/86
* revision 0.1 - 3/23/88  - bypass embedded quotated strings during search

#ifdef unix
	include 'ferret_cmn/ferret.parm'
#else
	INCLUDE 'FERRET_CMN:FERRET.PARM'
#endif

* calling argument declarations:
	CHARACTER*(*)	string
	CHARACTER*1	atom

* internal variable declarations:
	INTEGER		pos, slen

* initialize
	slen = LEN( string )
	pos  = 1

* loop top
 100	CONTINUE

* quotated string ?
	IF ( string(pos:pos) .EQ. '"' ) THEN
	   DO 110 pos = pos+1, slen
	      IF ( string(pos:pos) .EQ. '"' ) GOTO 200
 110	   CONTINUE
* ... unclosed quotation mark - flag it as a syntax error
	   ATOM_POS = atom_illegal_pos
	   RETURN

* found it ?
	ELSEIF ( string(pos:pos) .EQ. atom ) THEN
	   IF ( pos .EQ. slen ) THEN
	      ATOM_POS = atom_illegal_pos
	   ELSE
	      ATOM_POS = pos
	   ENDIF
	   RETURN
	ENDIF

* keep on looking ?
 200	IF ( pos .LT. slen ) THEN
	   pos = pos + 1
	   GOTO 100
	ENDIF

* no joy
	ATOM_POS = atom_not_found
	RETURN
	END
