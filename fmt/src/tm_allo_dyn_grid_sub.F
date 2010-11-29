	SUBROUTINE TM_allo_dyn_grid_sub( hook, grid, status )

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
* Allocate a grid from the dynamic grid area (above max_grids)

* The grids in xtm_grid.cmn fall into three categories (ideally they could
* and should all be category 3, but the development was historical)
* category 1: explicit (static) grids
*	these are located in the low indices of the grid common
*	(e.g. grid_name(3)).  They are created by READ_GRIDFILE, CD_*_GRIDS,
*	and by DEFINE GRID inside Ferret.  These are the grids that are
*	returned by TM_FIND_GRID_SLOT.
* category 2: temporary grids
*	these are located beginning at max_grids-1 and descending.  They have
*	only very temporary existence during the GET_UVAR_GRID process -- as
*	holding areas while dynamic grids (category 3) are built.
* category 3: dynamic grids
*	these are in the upper indices: above max_grids.  They are managed
*	efficiently with linked lists and usage counts.  They are used by
*	Ferret to hold "implicit" grids -- implied by expressions such as
*	SALT[GX=U] (the SALT grid with its X axis replaced from the U grid).
*	They are deleted as necessary to ensure that new grid definitions and
*	changes in user variable definitions do not corrupt memory. 

* Steve Hankin  NOAA/PMEL/TMAP

* 9/13/95 for Ferret version 4.2
* V510 *sh* added "hook" arg so this can be used for tmp grids, too

* Include files
#ifdef unix
	include 'tmap_dims.parm'
#include "tmap_dset.parm"
#include "gt_lib.parm"
	include 'tmap_errors.parm'
	include 'xtm_grid.cmn_text'
	include 'xio.cmn_text'
	external xgt_grid_data
#else
	INCLUDE 'TMAP_FORMAT:TMAP_DIMS.PARM'
	INCLUDE 'TMAP_FORMAT:TMAP_DSET.PARM'
	INCLUDE 'TMAP_FORMAT:GT_LIB.PARM'
	INCLUDE 'TMAP_FORMAT:TMAP_ERRORS.PARM'
	INCLUDE 'TMAP_FORMAT:XTM_GRID.CMN'
	INCLUDE 'TMAP_FORMAT:XIO.CMN_TEXT'
#endif

* Argument definitions
	INTEGER hook, grid, status

* Local definitions
	INTEGER next_free_grid

* take a grid from the "free" list and put it at the start of the "used" list
	IF ( grid_free_ptr .NE. 0 ) THEN
* ... unhook from free list
	   grid			= grid_free_ptr
	   next_free_grid  	= grid_flink(grid)
	   grid_free_ptr	= next_free_grid
* ... hook onto front of used list
	   grid_flink(grid)	= grid_flink(hook)
	   grid_blink(grid)	= hook
	   grid_flink(hook)= grid
	   grid_blink(grid_flink(grid))	= grid
	ELSE
	   CALL TM_ERRMSG (merr_gridlim, status,
     .                  'TM_ALLO_DYN_GRID', no_descfile, no_stepfile,
     .                  no_errstring, no_errstring, *9999)
	ENDIF

* signal that it is in use by a single claimant
	grid_use_cnt(grid) = 1

* successful completion
	status = merr_ok
 9999	RETURN

	END
