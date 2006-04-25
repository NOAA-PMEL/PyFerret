	INTEGER FUNCTION TM_DEALLO_DYN_LINE_SUB( line )

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
* Deallocate a line from the dynamic line area (above max_lines)
* Called quasi-recursively by tm_deallo_dyn_line

* Steve Hankin  NOAA/PMEL/TMAP

* 10/3/95 for Ferret version 4.2
* V510 *sh( 4/00 - allow deallocation of non-dynamic lines
* V600 *acm* 10/05 Call PACK_LINE_STORAGE for irregular axes to reset coord storage.

* Include files

	include 'tmap_dims.parm'
#include "tmap_dset.parm"
#include "gt_lib.parm"
	include 'tmap_errors.parm'
	include 'xtm_grid.cmn_text'
	external xgt_grid_data
	include 'xio.cmn_text'

* Argument definitions
	INTEGER line

* Local definitions
	INTEGER next_used_line

* initialize
	TM_DEALLO_DYN_LINE_SUB = 0

* screen out garbage (may come from temporary grids)
	IF ( line .LT. 1 .OR. line .GT. max_dyn_lines ) RETURN

* decrement the used count for this line
	line_use_cnt(line) = line_use_cnt(line) - 1
	IF ( line_use_cnt(line) .LT. 0 ) line_use_cnt(line) = 0

* if not a dynamic line or it is a "keeper" then we are done
	IF ( line.LE.max_lines .OR. line_keep_flag(line) ) THEN
	   RETURN
	ENDIF

* does anyone still have a claim on this line?
	IF ( line_use_cnt(line) .GT. 0 ) THEN
	   RETURN

* a use count of -1 results from a line which was allocated and then promptly
* deallocated without being used
	ELSEIF ( line_use_cnt(line) .EQ. 0 ) THEN

	   line_name(line) = char_init16

* take this line from the "used" list and put it on the "free" list
* ... hook onto free list
	   next_used_line   = line_flink(line)
	   line_flink(line) = line_free_ptr  
	   line_free_ptr    = line
	   
* ... unhook from used list
	   line_flink(line_blink(line)) = next_used_line
	   line_blink(next_used_line)   = line_blink(line)

	ELSE
* internal error -- the use count is negative!!
	   CALL TM_NOTE('Intern err: TM_DEALLO_DYN_LINE:2 !!!', lunit_errors)
	ENDIF

* Recover the coordinate storage used for an irregularly-spaced line.
        IF ( .NOT. line_regular(line) ) CALL PACK_LINE_STORAGE (line)

* return flag to indicate if the quasi-recursive descent ends here
* ... zero or index of parent axis to deallocate
	TM_DEALLO_DYN_LINE_SUB = line_parent(line)

* successful completion
	RETURN

	END
