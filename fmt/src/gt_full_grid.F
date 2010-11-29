	INTEGER FUNCTION GT_FULL_GRID ( grid_num, idim )

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
* find the full range of the indicated grid on the indicated axis in SUBSCRIPT
* values

* programmer - mark verschell (from GRID_SUBSCRIPT_EXTREMES by steve hankin)
* NOAA/PMEL, Seattle, WA - Tropical Modeling and Analysis Program
* written for VAX computer under VMS operating system
*
* revision 0.00 - 04/23/87
* revision 0.10 - 07/19/88 - Simplified grid/line location in common
* revision 0.20 - 11/29/88 - Unknown/normal line check done on grid_line now
*			     and removed references to munknown, mnormal
*
* include files
#ifdef unix
#include "gt_lib.parm"
	include 'tmap_dims.parm'
        include 'xtm_grid.cmn_text'
        external xgt_grid_data
#else
	INCLUDE	'TMAP_FORMAT:TMAP_DIMS.PARM'
	INCLUDE	'TMAP_FORMAT:GT_LIB.PARM'
	INCLUDE	'TMAP_FORMAT:XTM_GRID.CMN'
#endif

* calling argument declarations
	INTEGER		grid_num, idim

* local variable declarations
	INTEGER		line_len, line_num, nline

* locate line in common storage
	line_num = grid_line( idim, grid_num )
	line_len = line_dim( line_num )

	IF ( line_num .EQ. mpsunkn .OR. line_num .EQ. mpsnorm) THEN
	  GT_FULL_GRID = 1
	ELSE
	  GT_FULL_GRID = line_len
	ENDIF

	RETURN
	END
