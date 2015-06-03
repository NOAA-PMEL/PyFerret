	SUBROUTINE TM_deallo_dyn_grid( grid )

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
* Deallocate a grid from the dynamic grid area (above max_grids)

* Steve Hankin  NOAA/PMEL/TMAP

* 9/13/95 for Ferret version 4.2
* V510 *sh* 4/00 - remove actual deallocation of grid to tm_deallo_dyn_grid_sub
*		 - this routine can now be called for static grids, too
*		 - deallocate lines only for dynamic grids, since static grids
*		   do not get deleted when their use count falls to zero
* V674 2/12 *acm* 6D Ferret: use nferdims rather than 4 for dimension indexing

* Include files
	include 'tmap_dims.parm'
#include "tmap_dset.parm"
#include "gt_lib.parm"
	include 'tmap_errors.parm'
	include 'xtm_grid.cmn_text'
		include 'xio.cmn_text'

* Argument definitions
	INTEGER grid

* Local definitions
	INTEGER idim

* deallocate the grid, itself
	CALL TM_DEALLO_DYN_GRID_SUB(grid)

* if the grid was actually removed, then also deallocate the axes that it used
* Check first that the grid number is in range

	IF ( grid .GT. max_grids) THEN
        IF ( grid_use_cnt(grid) .EQ. 0 ) THEN

*release claim to any dynamic lines this may use
	   DO 100 idim = 1, nferdims
	      CALL TM_DEALLO_DYN_LINE( grid_line(idim,grid))
 100	   CONTINUE

	ENDIF
	ENDIF

* successful completion
	RETURN

	END
