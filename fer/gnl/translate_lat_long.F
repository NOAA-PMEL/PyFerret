	SUBROUTINE TRANSLATE_LAT_LONG(string, idim, answer, formatted, status)

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
* a floating point value using special format rules

* programmer - steve hankin
* NOAA/PMEL, Seattle, WA - Tropical Modeling and Analysis Program
* written for VAX computer under VMS operating system
*
* revision 0.0 - 4/11/86
* revision 0.1 - 3/14/89 - return "formatted" flag
* V300  9/93 - insist that "N", "S", "E", "W" are the LAST character
* V420 (IBM/AIX port of V411): 10/5/95 - buffer "READ(var,*"

* possible formats:
*   x coord:  #  or #E  or  #W  for East  or West,  respectively
*   y coord:  #  or #S  or  #N  for South or North, respectively
*   z coord:  #
*   t coord:  #  or dd-mmm-yyyy@hh-mm-ss  eg 14-JAN-1986@11:15

#ifdef unix
	include 'ferret_cmn/errmsg.parm'
	include 'ferret_cmn/ferret.parm'
#else
	INCLUDE 'FERRET_CMN:ERRMSG.PARM'
	INCLUDE 'FERRET_CMN:FERRET.PARM'
#endif

	CHARACTER*(*)	string
	LOGICAL		n_given, s_given, e_given, w_given, formatted
	INTEGER		n_pos, s_pos, e_pos, w_pos, slen, status, idim
	REAL*8		answer

* local declarations
	CHARACTER	buff*32		! 10/95

* initialize
	slen = LEN ( string )
	n_given = string(slen:slen).EQ.'N' .OR. string(slen:slen).EQ.'n'
	s_given = string(slen:slen).EQ.'S' .OR. string(slen:slen).EQ.'s'
	e_given = string(slen:slen).EQ.'E' .OR. string(slen:slen).EQ.'e'
	w_given = string(slen:slen).EQ.'W' .OR. string(slen:slen).EQ.'w'
	formatted = n_given .OR. s_given .OR. e_given .OR. w_given

* how much of the string is numeric ?
	IF ( formatted ) slen = slen - 1

* decode the value
	buff =  string(1:slen)		! 10/95
	READ ( buff, * , ERR=5000 ) answer

* make corrections for latitude/longitude notation
	IF     ( n_given ) THEN

* latN - no change
	   IF ( idim .NE. y_dim ) GOTO 5000
	ELSEIF ( s_given ) THEN

* latS - make negative
	   IF ( idim .NE. y_dim ) GOTO 5000
	   answer = -answer
	ELSEIF ( e_given ) THEN

* longE - no change
	   IF ( idim .NE. x_dim ) GOTO 5000
	ELSEIF ( w_given ) THEN

* longW - convert	
	   IF ( idim .NE. x_dim ) GOTO 5000
	   answer = 360. - answer
	ENDIF

	status = ferr_ok
	RETURN

* error exit
 5000	CALL ERRMSG	( ferr_syntax, status, string, *5010 )
 5010	RETURN
	END
