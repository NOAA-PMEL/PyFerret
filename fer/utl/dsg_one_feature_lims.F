	SUBROUTINE DSG_ONE_FEATURE_LIMS(dset, base, flen, ifeature,
     .					lolims, hilims)

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
* Compute the lo/hi coordinate limits of the given feature of this dsg dataset
* For its "obs" coordinate axes.  (e.g. for a time series collection time
* is an obs axis, whereas lon and lat are instance axes

* NOAA/PMEL, Seattle, WA - Tropical Modeling and Analysis Program

* V720 4/17 *sh*

	include 'tmap_dims.parm'
	include 'ferret.parm'
	include 'errmsg.parm'
	include 'xtm_grid.cmn_text'
	include 'xdset_info.cmn_text'
	include 'xdyn_linemem.cmn_text'
	include 'xcontext.cmn'
	include 'xdsg_context.cmn'
	include 'xprog_state.cmn'
	include 'xvariables.cmn'

* calling argument declarations:
	INTEGER dset, base, flen,ifeature
	REAL lolims(4), hilims(4)

* internal variable declarations:
        INTEGER i, lm, idim, var
	REAL cmin, cmax, coord

! DSGTBD: It is inefficient to re-compute the extremes time after time
! see discussion in dsg_feature_lims

* get the coordinate extremes of this feature
	DO idim = 1, 4
	   IF (cxdsg_has_coord(idim)) THEN
	      lm = cxdsg_coord_lm(idim)
	      IF (cxdsg_is_obs_coord(idim)) THEN
	         cmin = arbitrary_large_val8
	         cmax = arbitrary_small_val8
	         DO i = base+1, base+flen
	            coord = dsg_linemem(lm)%ptr(i)
	            IF (coord .LT. cmin) cmin = coord
	            IF (coord .GT. cmax) cmax = coord
	         ENDDO
	         lolims(idim) = cmin
	         hilims(idim) = cmax
	      ELSE
	         coord = dsg_linemem(lm)%ptr(ifeature)
	         lolims(idim) = coord
	         hilims(idim) = coord
	      ENDIF
	   ELSE
	      lolims(idim) = unspecified_val8
	      hilims(idim) = unspecified_val8
	   ENDIF
	ENDDO

	RETURN
	END
