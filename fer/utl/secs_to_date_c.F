	SUBROUTINE secs_to_date_c( bcsecs, hdate )

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
* C-callable routine to convert the number of seconds since BC to a date string

* programmer - steve hankin
* NOAA/PMEL, Seattle, WA - Tropical Modeling and Analysis Program

* v5.33 *acm* SECS_TO_DATE needs arg cal_id. Use Gregorian - these calls
*             are used in converting today's date for labels, etc.  See
*             gui/Utility.c and gui/JC_Utility.c

* V400: 11/94 

        IMPLICIT NONE
	include	'tmap_dims.parm'
	include	'ferret.parm'
	include 'xprog_state.cmn'

* calling argument declarations:
	REAL*8 bcsecs
        INTEGER*1 hdate(*)

* local variable declarations
	INTEGER	  TM_LENSTR1, slen, cal_id
	CHARACTER SECS_TO_DATE*20, temp_str*20

	cal_id = 1	! Gregorian (default) calendar

* do the conversion using FORTRAN CHARACTER variables
	temp_str = SECS_TO_DATE(bcsecs, cal_id)

* now convert the result into a C string
	slen = TM_LENSTR1( temp_str )
	CALL TM_FTOC_STRNG( temp_str, hdate, slen )

	RETURN
	END

