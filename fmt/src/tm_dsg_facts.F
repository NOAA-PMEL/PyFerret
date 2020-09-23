	SUBROUTINE TM_DSG_FACTS( grid, orientation, obsdimlen, 
     .			feature_line, is_dsg, is_cmpnd )

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
* return the feature orientation and netCDF dim length of the observations
* in a DSF file
* THIS ROUTINE SHOULD ONLY BE CALLED FOR GRIDS KNOWN TO BE DSG 

* V720 *sh* 4/17
* V74+ *acm* 9/18 Allow an abstract axis (e.g. merging grids)
* V761 *acm* 7/20 Return a flag for a compound type, timeseriesProfile or
*                 trajectoryProfile.
* V763 *acm* 9/20 Return if grid not defined.

      include 'netcdf.inc'
      include 'tmap_errors.parm'
      include 'tmap_dims.parm'
      include 'implicit.parm'
#include "gt_lib.parm"
#include "tmap_dset.parm"
      include 'xtm_grid.cmn_text'
      include 'xdset_info.cmn_text'
      include 'xfr_grid.cmn'

* argument declarations
	LOGICAL is_dsg, is_cmpnd
	INTEGER grid, orientation, obsdimlen, feature_line

* internal variable declarations
	INTEGER DSG_WHATS_IT, e_line, dset, mabstract, status

	is_dsg = .FALSE.
	is_cmpnd = .FALSE.

	IF (grid .LE. 0) RETURN

* the E axis points to the DSG orientation. 
* Or is this even a dsg grid? The dataset may be dsg, but what if we're just 
* defining some other variable, maybe defining a mask.

	e_line = grid_line(e_dim,grid)
	   
	mabstract = grid_line( x_dim, mgrid_abstract )

	IF (e_line.EQ.mpsnorm .OR. e_line.EQ.mabstract) GOTO 9999

	dset = line_dsg_dset(e_line)

* Can happen with a use/feature=none

	IF (dset.EQ. pline_not_dsg) GOTO 9999

	is_dsg = .TRUE.
	orientation = dsg_orientation(dset)
	feature_line = grid_line(orientation,grid)
	obsdimlen = dsg_obs_dimlen(dset)

	is_cmpnd = orientation.EQ.pfeatureType_TrajectoryProfile .OR. 
     .		   orientation.EQ.pfeatureType_TimeseriesProfile

* success
        RETURN

* unreported error -- just a message
 9999	RETURN

        END








