	SUBROUTINE TM_TRANSLATE_LAT_LONG (string,idim,answer,status)
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
*
* convert an ascii representation of a latitude or longitude value into 
* a floating point value using special format rules
*
* possible formats:
*   x coord:  #  or #E  or  #W  for East  or West,  respectively
*   y coord:  #  or #S  or  #N  for South or North, respectively
*   z coord:  #
*   t coord:  #  or dd-mmm-yyyy@hh-mm-ss  eg 14-JAN-1986@11:15
*
* programmer - Mark Verschell
* NOAA/PMEL, Seattle, WA - Tropical Modeling and Analysis Program
* written for VAX computer under VMS operating system
*
* revision 0.00 - 12/14/88 - from program by Steve Hankin
*
* include files
#ifdef unix
	include 'tmap_errors.parm'
#include "gt_lib.parm"
#else
	INCLUDE 'TMAP_FORMAT:TMAP_ERRORS.PARM'
	INCLUDE 'TMAP_FORMAT:GT_LIB.PARM'
#endif

	CHARACTER*(*)	string
	LOGICAL		n_given, s_given, e_given, w_given
	INTEGER		n_pos, s_pos, e_pos, w_pos, end_pos, status, idim
	REAL*8		answer

* initialize
	n_pos = MAX ( INDEX(string,'N'), INDEX(string,'n') )
	s_pos = MAX ( INDEX(string,'S'), INDEX(string,'s') )
	e_pos = MAX ( INDEX(string,'E'), INDEX(string,'e') )
	w_pos = MAX ( INDEX(string,'W'), INDEX(string,'w') )

	n_given = n_pos .GT. 0
	s_given = s_pos .GT. 0
	e_given = e_pos .GT. 0
	w_given = w_pos .GT. 0

* how much of the string is numeric ?
	end_pos = LEN ( string )
	IF ( n_given.OR.s_given.OR.e_given.OR.w_given )	end_pos = end_pos - 1

* decode the value
	READ ( string(1:end_pos), * , ERR=9000 ) answer

* make corrections for latitude/longitude notation
	IF  ( n_given ) THEN

* latN - no change
	  IF ( idim .NE. y_dim ) GOTO 9000
	ELSEIF ( s_given ) THEN

* latS - make negative
	  IF ( idim .NE. y_dim ) GOTO 9000
	  answer = -answer
	ELSEIF ( e_given ) THEN

* longE - no change
	  IF ( idim .NE. x_dim ) GOTO 9000
	ELSEIF ( w_given ) THEN

* longW - convert	
	  IF ( idim .NE. x_dim ) GOTO 9000
	  answer = 360. - answer
	ENDIF

	status = merr_ok
	GOTO 9990

* errors
 9000	CALL TM_ERRMSG ( merr_syntax, status, 'TM_TRANSLATE_LAT_LONG',
     .	                 no_descfile, no_stepfile, 
     .			 string, no_errstring, *9990 )

 9990	RETURN
	END
