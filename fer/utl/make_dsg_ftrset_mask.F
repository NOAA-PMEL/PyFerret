	SUBROUTINE MAKE_DSG_FTRSET_MASK(dset, cx, nfeatures, fmask, 
     .                                  nftrsets, smask )
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
* Create a timeseries-station or trajectory mask for a DSG based upon
* world coordinate limits and/or a user-defined mask.  Uses a feature-mask
* defined by a call to MAKE_DSG_FEATURE_MASK.

* NOAA/PMEL, Seattle, WA - SDIG

* V762 8/20 *ACM*
* V762 9/20 *ACM* for both TimeseriesProfile and TrajectoryProfile data types

	include 'tmap_dims.parm'
	include 'ferret.parm'
	include 'xtm_grid.cmn_text'
	include 'xdset_info.cmn_text'
	include 'xdyn_linemem.cmn_text'

#include "tmap_dset.parm"

* calling argument declarations:
	INTEGER dset, cx, nfeatures, nftrsets
	LOGICAL fmask(nfeatures), smask(*)

* internal variable declarations:
        LOGICAL its_dsg, its_cmpnd
	INTEGER grid, i, lm, orientation, obsdimlen, feature_line, 
     .		lm_index

        INTEGER, DIMENSION(:), ALLOCATABLE :: station_index

* Initialize
	DO i = 1, nftrsets
	   smask(i) = .FALSE.
	ENDDO

* Get dataset info

	grid = dsg_xlate_grid(dset)
	CALL TM_DSG_FACTS( grid, orientation, obsdimlen, feature_line, 
     .                     its_dsg, its_cmpnd )

	IF (.NOT. its_cmpnd) RETURN

* Get the station-index and use it to pick out the station-id or trajectory-id mask
* from the feature mask

	ALLOCATE (station_index(nfeatures))
	lm_index = dsg_loaded_lm(dsg_index_var(dset))

	DO i = 1, nfeatures
	   station_index(i) = dsg_linemem(lm_index)%ptr(i) + 1
	ENDDO

* which station/traj are represented in the feature mask

	DO i = 1, nfeatures
	   IF (fmask(i)) smask(station_index(i)) = .TRUE.
	ENDDO

	DEALLOCATE(station_index)

	RETURN
	END
