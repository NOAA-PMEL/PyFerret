	SUBROUTINE CHECK_FORMAT( string, status )

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
* Check a FORTRAN FORMAT string for superficial syntax errors

* programmer - steve hankin
* NOAA/PMEL, Seattle, WA - Tropical Modeling and Analysis Program

* V301 1/94 - extracted from EQUAL_FORMAT
* V320 8/94 *kob* -IBM port - had to use risc buff for string concats
*       *acm* 3/12 cleanup ifdefs and unnecessary include files
* V750 3/18 *acm* issue 554, explain what formasts are allowed.
* V750 *acm*  3/18 Issue 950: Allowing simplified I formats 

        IMPLICIT NONE
* calling argument declarations
	INTEGER		status
	CHARACTER*(*)	string

* internal variable declarations
	INTEGER	left_paren, right_paren, integer_form

	include 'ferret.parm'
	include 'errmsg.parm'
	include 'xrisc.cmn'

* check out user-specified format
	left_paren  = INDEX( string, '(' )
	right_paren = INDEX( string, ')' )
	IF ( left_paren .EQ. 0 
     .	.OR. left_paren .GE. right_paren ) GOTO 5040

* success
	status = ferr_ok
	RETURN

* syntax error
 5000	RETURN
 5040   risc_buff = string
	CALL ERRMSG( ferr_syntax, status,
     .  'Unknown format or format need parentheses'//pCR//
     .  'Valid formats are CDF, UNFORMATTED, STREAM, COMMA, TAB or limited Fortran formats'//pCR//
     .			risc_buff, *5000 )
	END
