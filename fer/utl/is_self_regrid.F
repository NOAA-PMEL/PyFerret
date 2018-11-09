	LOGICAL FUNCTION IS_SELF_REGRID (idim, src_grid, dst_grid, trans)


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
* 5/15/2018 
* Ansley Manke 5/2018
* V741  5/18 *acm* issue 1868: gridding transforms that will grid to same axis
* V741  6/18 *acm* Also check if the axis has the same definition, though its name 
*                  may be different, e.g. TIME axis is one of the climatology axes.
*
*  Return a flag, true if the current operation is a regridding-to
*  the source axis. See regrid_to_self.F 

        IMPLICIT NONE
	include 'tmap_dims.parm'
	include	'xtm_grid.cmn_text'
	external xgt_grid_data
	include	'ferret.parm'

* calling argument declaratinos
	INTEGER idim, src_grid, dst_grid, trans

* Local declarations
	LOGICAL REGRID_TO_SELF, TM_SAME_LINE_DEF, same

	INTEGER src_line, dst_line

	src_line = grid_line (idim, src_grid)
	dst_line = grid_line (idim, dst_grid)
	same = src_line.EQ.dst_line .OR. TM_SAME_LINE_DEF(src_line, dst_line)
	IS_SELF_REGRID = (same .AND. REGRID_TO_SELF (trans) )


	RETURN
	END
