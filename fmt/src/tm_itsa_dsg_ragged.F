	LOGICAL FUNCTION TM_ITSA_DSG_RAGGED ( grid )

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
* determine if this grid represents a contiguous ragged array
* To be so it must have both an E axis and a obs axis

* V720 *sh* 4/17

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
	INTEGER line, dset, orient

* DEFAULT
	TM_ITSA_DSG_RAGGED = .FALSE.

* To be a DSG contiguous ragged grid it must be 2D with
* the E axis pointing to the DSG orientation and the
* oriented axis carrying the true netCDF obs dim length
* TrajectoryProfile or TimeseriesProfile have obs in z

	line = grid_line(e_dim,grid)
	IF (line .EQ. mpsnorm) RETURN

	dset = line_dsg_dset(line)

	IF (dset .EQ. pline_not_dsg) THEN
	   TM_ITSA_DSG_RAGGED = .FALSE.
	ELSE

	  orient = dsg_orientation(dset)
	  IF  (orient.EQ.y_dim .OR. orient.EQ.f_dim) orient = z_dim

	   line = grid_line(orient,grid)
	   IF (line .EQ. mpsnorm) RETURN

* if the orientation axis also points to the dset, this is a contiguous
* ragged DSG feature colletion
	   TM_ITSA_DSG_RAGGED = line_dsg_dset(line) .EQ. dset
	ENDIF
        RETURN

        END
