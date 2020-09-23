	SUBROUTINE DSG_FEATURE_LIMS(dset, nfeatures, lolims, hilims)

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
* Compute the lo/hi coordinate limits of each feature of the given dsg dataset
* For its "obs" coordinate axes.  (e.g. for a time series collection time
* is an obs axis, whereas lon and lat are instance axes

* NOAA/PMEL, Seattle, WA - Tropical Modeling and Analysis Program

* V720 4/17 *sh*   
* V741 6/18 *acm* Go ahead and get single-value coordinates from feature-level vars
* V751 6/19 *acm* Set limits correctly when data in the entire feature is missing

	include 'tmap_dims.parm'
	include 'ferret.parm'
	include 'errmsg.parm'
	include 'xtm_grid.cmn_text'
	include 'xdset_info.cmn_text'
	include 'xdyn_linemem.cmn_text'
	include 'xcontext.cmn'
	include 'xdsg_context.cmn'
	include 'xvariables.cmn'
#include "tmap_dset.parm"

* calling argument declarations:
	INTEGER dset, nfeatures
	REAL lolims(4,nfeatures), hilims(4,nfeatures)

* internal variable declarations:
        LOGICAL tsprof
	INTEGER i, j, lm, idim, var, rslm, obs_dimlen, flen, iobs0
	REAL cmin, cmax, coord, bad

        INTEGER, DIMENSION(:), ALLOCATABLE :: station_index

* initialize
	obs_dimlen = dsg_obs_dimlen(dset)
	rslm = dsg_loaded_lm(dsg_row_size_var(dset))

! DSGTBD: It is inefficient to re-compute the extremes time after time
! A good solution to this would be
!  1. check for attributes '_feature_mins_lm' and '_feature_maxs_lm' on var
!  2a. if they do not exist, then
!    i. compute min and max as below
!    ii. allocate dsg_linemem sufficient to store them -- getting the lm indices
!    iii. create the attributes (hidden atts if possible), attaching the lms
!  2a. if they do exist, then
!    i. get the lm values from the attributes
!    ii. copy the stored extremes back from lm memory
!  3.  Remember to free this memory when closing the DSG dataset


* For TimeseriesProfile, get the station-index and use it to set feature-coordinate using station-location


	tsprof = (dsg_orientation(dset) .EQ. pfeatureType_TimeseriesProfile) 
	IF (tsprof) THEN

	   ALLOCATE (station_index(nfeatures))
	   lm = dsg_loaded_lm(dsg_index_var(dset))

	   DO i = 1, nfeatures
	      station_index(i) = dsg_linemem(lm)%ptr(i) + 1
	    ENDDO
	    
	ENDIF


* get the DSG coordinate extremes

	lolims = unspecified_val8
	hilims = unspecified_val8

	DO idim = 1, 4
	   bad = cxdsg_bad_val(idim)
	   IF (cxdsg_has_coord(idim)) THEN
	      lm = cxdsg_coord_lm(idim)
	      IF (cxdsg_is_obs_coord(idim)) THEN
	         iobs0 = 0
	         DO i = 1, nfeatures
	            flen = dsg_linemem(rslm)%ptr(i)   ! feature length
	            cmin = arbitrary_large_val8
	            cmax = arbitrary_small_val8
	            DO j = iobs0+1, iobs0+flen
	               coord = dsg_linemem(lm)%ptr(j)
		       IF (coord .EQ. bad) CYCLE
	               IF (coord .LT. cmin) cmin = coord
	               IF (coord .GT. cmax) cmax = coord
	            ENDDO
	            IF (cmin .LT. arbitrary_large_val8) lolims(idim,i) = cmin
	            IF (cmax .GT. arbitrary_small_val8) hilims(idim,i) = cmax
	            iobs0 = iobs0 + flen
	         ENDDO
* The single coordinate for this feature
	      ELSE

* location of profiles (maybe at a station) 

	         DO i = 1, nfeatures
	            IF (tsprof) THEN
		       coord = dsg_linemem(lm)%ptr(station_index(i))
		    ELSE
		       coord = dsg_linemem(lm)%ptr(i)
		    ENDIF
		    IF (coord .EQ. bad) CYCLE
	            lolims(idim,i) = coord
	            hilims(idim,i) = coord
	         ENDDO
	      ENDIF
	   ELSE
	      DO i = 1, nfeatures
	         lolims(idim,i) = unspecified_val8
	         hilims(idim,i) = unspecified_val8
	      ENDDO
	   ENDIF
	ENDDO

	IF (tsprof) DEALLOCATE(station_index)

	RETURN
	END
