	SUBROUTINE TM_TRANSLATE_AXIS (string, idim, answer, status)
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
* convert an ascii representation of an axis limit into a floating point 
* representation using special format rules
*
* programmer - Mark Verschell
* NOAA/PMEL, Seattle, WA - Tropical Modeling and Analysis Program
* written for VAX computer under VMS operating system
*
* possible formats for world coordinate values:
*   x coord:  #  or #E  or  #W  for East, East, and West respectively
*   y coord:  #  or #S  or  #N  for South, South, and North respectively
*   z coord:  #
*   t coord:  #  or dd-mmm-yyyy@hh:mm:ss  eg 14-JAN-1986@11:15
*
* revision 0.00 - 12/13/88 - from TRANSLATE_LIMITS by Steve Hankin
*
* Include files
#ifdef unix
	include 'tmap_format/tmap_errors.parm'
	include 'tmap_format/gt_lib.parm'
#else
	INCLUDE 'TMAP_FORMAT:TMAP_ERRORS.PARM'
	INCLUDE 'TMAP_FORMAT:GT_LIB.PARM'
#endif

* calling argument declarations:
	CHARACTER*(*)	string
	INTEGER		status, idim
	REAL*8		answer

* internal variable declarations:
	INTEGER*4	end_pos, dy, mnth, yr, hr, mnt, scnd
	INTEGER*4	TM_LENSTR1
	REAL*8		TM_SECS_FROM_BC

* internal parameter declarations:
	CHARACTER*1	tab
	PARAMETER     ( tab = CHAR(9) )

	end_pos = TM_LENSTR1( string )

	IF ( idim .NE. t_dim ) THEN
* /X /Y or /Z
	  CALL TM_TRANSLATE_LAT_LONG ( string(1:end_pos), idim, answer,
     .	                               status )
	  IF ( status .NE. merr_ok ) GOTO 9990		! already reported

	ELSE
* /T
* ... is it a timestep?
	  READ (string(1:end_pos),*,ERR=200) answer
	  GOTO 1000
* ... is it a date/time string? (store as seconds for further processing)
  200	  CALL TM_BREAK_DATE(string,dy,mnth,yr,hr,mnt,scnd,status)
	  IF ( status .NE. merr_ok ) GOTO 9990 		! Report in call routine
	  answer = -1.D0*TM_SECS_FROM_BC(dy,mnth,yr,hr,mnt,scnd,status)
	ENDIF

* successful completion
 1000	status = merr_ok

 9990	RETURN
	END
