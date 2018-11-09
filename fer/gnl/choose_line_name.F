	SUBROUTINE CHOOSE_LINE_NAME (axis, original, axname)
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
* For the given axis number, return either line_name(axis) or
* line_name_orig(axis).  When mode_upcase is set, return the line_name
* version which has been upper-cased. 

* If mode_upcase is false, atempt to return line_name_orig.
* -- First check whether line_name_orig is defined. 
* -- If the lengths of line_name and line_name_orig differ, the line name was changed 
*    by Ferret for uniqueness. In these cases, check the value of the flag "original". 
* -- If mode_upcase is false, and original is true, then we want the original form 
*    no matter what. E.g. for xml output, we want the original axis name from the file.
*    For other cases, such as a generic SHOW GRID, we want the name in use for the 
*    Ferret session.
* V698 2/16 *acm*  For ticket 1786: if mode_upcase is canceled write axis name
*                  using original upper/lowercase spelling


        IMPLICIT NONE
	include 'tmap_dims.parm'
	include 'xtm_grid.cmn_text'
#	include "tmap_dset.parm"
	include 'xprog_state.cmn'

	LOGICAL original 
	INTEGER TM_LENSTR, axis
	CHARACTER*(*) axname

	axname = line_name(axis)

	IF (.NOT. mode_upcase_output) THEN

* If there is no good line_name_orig defined, keep line_name
	   IF (TM_LENSTR(line_name_orig(axis)).EQ.0  .OR.
     .         line_name_orig(axis)(1:2) .EQ. char_init) THEN
             axname = line_name(axis)

* If the axis has been renamed for uniqueness, return line_name
* unless original has been set.

	   ELSE IF (TM_LENSTR(line_name_orig(axis)) .NE. 
     .              TM_LENSTR(line_name(axis)     ) .AND. .NOT.original) THEN
             axname = line_name(axis)

* Otherwise if upcase is turned off, return line_name_orig
           ELSE
	      axname = line_name_orig(axis)
           ENDIF
	ENDIF

	
	RETURN
	END
