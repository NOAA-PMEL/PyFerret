	SUBROUTINE TM_GET_LIKE_LINE ( newline, newname, linenum, status)
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
* Will compare "newline" to other line definitions in common. If there is a
* match then "linenum" returns with the number of the match. If there is no
* match, then a new slot is located and the name "newname" is used to name it,
* and "linenum" returns with the number of the new slot.
*
* revision 0.00 - 11/22/88
* V541 set status to OK if we find a match.
*
* Arguments
	INTEGER*4	newline, linenum, status
	CHARACTER*(*)	newname
*
* INCLUDE FILES
	include 'tmap_dims.parm'
#include "tmap_dset.parm"
#include "gt_lib.parm"
	include 'tmap_errors.parm'
	include 'xtm_grid.cmn_text'
	external xgt_grid_data
*
* Local definitions
	INTEGER*4	line_match
	INTEGER*4	TM_FIND_LIKE_LINE, TM_FIND_LINE_SLOT
*
* Check for match
	line_match = TM_FIND_LIKE_LINE ( newline )
	IF (line_match .NE. unspecified_int4) THEN
	  linenum = line_match
	  status = merr_ok
	  GOTO 9000
	ENDIF
*
* No match, find empty slot for storage
	status = TM_FIND_LINE_SLOT ( linenum )
	IF (status .NE. merr_ok) GOTO 9000
*
* Move everything over
	CALL TM_COPY_LINE ( newline, linenum )
*
* Name the new line
	line_name(linenum) = newname
*
	status = merr_ok
 9000	RETURN
	END
