	INTEGER FUNCTION TM_DSG_DSET_FROM_GRID(grid)

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
* return the dataset that owns this DSG grid

* V720 *sh* 4/17
* V763 *acm* 9/20 Return if grid not defined.

      include 'netcdf.inc'
      include 'tmap_errors.parm'
      include 'tmap_dims.parm'
      include 'implicit.parm'
#include "gt_lib.parm"
      include 'xtm_grid.cmn_text'
      include 'xdset_info.cmn_text'

* argument declarations
	INTEGER grid

* internal variable declarations
	INTEGER e_line, status
	
	TM_DSG_DSET_FROM_GRID = pline_not_dsg
	IF (grid .LE. 0) RETURN

* every DSG grid has an E axis with line_dsg_dset
	e_line = grid_line(e_dim,grid)
	IF (e_line .EQ. mpsnorm) THEN
	   TM_DSG_DSET_FROM_GRID = pline_not_dsg
	ELSE
	   TM_DSG_DSET_FROM_GRID = line_dsg_dset(e_line)
	ENDIF

* success
        RETURN

        END








