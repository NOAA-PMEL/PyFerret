	SUBROUTINE TM_init_dyn_grids

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
* Initialize machinery for dynamic grids and lines

* Steve Hankin  NOAA/PMEL/TMAP
*
* 9/13/95 for Ferret version 4.2
* V510 *sh* 3/00 - initialize "hooks" for managing temporary grids and lines
*      *sh* 5/00 - set line_keep_flag everywhere to FALSE
* V570 *acm*5/04 - remove line_has_edges flag; never used.

* Include files
	include 'tmap_dims.parm'
#include "tmap_dset.parm"
#include "gt_lib.parm"
	include 'implicit.parm'
	include 'xtm_grid.cmn_text'
	external xgt_grid_data

* Argument definitions

* Local definitions
	INTEGER i


***** GRIDS *****
* grid(max_grids) is used as the chain start for the "in-use" grids
	grid_flink(max_grids) = max_grids	! points to self: empty chain

* grid(tmp_grid_hook) is used as the chain start for the "temporary" grids
	grid_flink(tmp_grid_hook) = tmp_grid_hook  ! points to self: empty

* no "dynamic" grids (above max_grids) are "used" (yet)
	DO 100 i = 1, max_dyn_grids
 100	grid_use_cnt(i) = 0

* all the "dynamic" grids (eq to or above max_grids) in the "free" linked list
	grid_free_ptr = max_grids+1
	DO 200 i = max_grids+1, max_dyn_grids
 200	grid_flink(i) = i+1
	grid_flink(max_dyn_grids) = 0	! end of chain

***** LINES *****
* line(max_lines) is used as the chain start for the "in-use" lines
	line_flink(max_lines) = max_lines	! points to self: empty chain

* line(tmp_line_hook) is used as the chain start for the "temporary" lines
	line_flink(tmp_line_hook) = tmp_line_hook  ! points to self: empty

* no "dynamic" lines (above max_lines) are "used" (yet)
	DO 300 i = 1, max_dyn_lines
	   line_use_cnt  (i) = 0
	   line_keep_flag(i) = .FALSE.
 300	CONTINUE
	DO 310 i = max_lines+1, max_dyn_lines
	   line_parent(i) = 0
	   line_class (i) = pline_class_basic
 310	CONTINUE

* all the "dynamic" lines (eq to or above max_lines) in the "free" linked list
	line_free_ptr = max_lines+1
	DO 400 i = max_lines+1, max_dyn_lines
 400	line_flink(i) = i+1
	line_flink(max_dyn_lines) = 0	! end of chain

	RETURN

	END


