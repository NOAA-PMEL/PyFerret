	SUBROUTINE TM_DATE_REFORMAT( indate,  inform,
     .				     outdate, outform, status)
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
* format outform

* *sh* 2/13/95
*
* Arguments:
*	indate	 - input date string
*	inform	 - code for input format
*	outdate	 - output date string
*	outform	 - code for output format
*	status   - returning status
*
	CHARACTER*(*)	indate, outdate
	INTEGER		inform, outform, status
*
* Inclusions
*
#ifdef unix
	include 'tmap_format/tmap_errors.parm'
#else
	INCLUDE 'TMAP_FORMAT:TMAP_ERRORS.PARM'
#endif

*
* Local definitions
*
	INTEGER	year, month, day, hour, minute, second, dlen

* Break up the input date string to it's various components
	CALL TM_BREAK_FMT_DATE ( inform, indate, year, month, day,
     .				    hour, minute, second, status )
	IF ( status .NE. merr_ok ) GOTO 9000

* and reassemble it into the desired output format
	CALL TM_MAKE_FMT_DATE ( outform, outdate, year, month, day,
     .				   hour, minute, second, dlen )

* Finished
	status = merr_ok
	RETURN

* Errors
 9000	RETURN
	END



