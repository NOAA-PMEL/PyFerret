	SUBROUTINE MAKE_DSG_OBS_MASK(dset, cx, ifeature, iobs0,
     .				     use_obs, nobs)

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
* Create an observation-level mask for a single feature of a DSG based upon
* world coordinate limits in the given context

* NOAA/PMEL, Seattle, WA - Tropical Modeling and Analysis Program

* V720 4/17 *sh*
* V741 6/18 *acm* Handle modulo longitudes
* TODO:  Should return the count of .TRUE. in the mask so the calling routine 
*        could skip applying the mask if count=nobs

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
#include "tmap_dset.parm"

* calling argument declarations:
	INTEGER dset, cx, ifeature, iobs0, nobs
	LOGICAL use_obs(nobs)

* internal variable declarations:
        INTEGER i, idim, lm, iptr
	REAL    coord, axmod, bad, val, del_lo, del_hi,
     .		constrain_lo_lo(4), constrain_hi_hi(4)

	axmod = 360.0 

! DSGTBD: presently the same constraints mask gets re-computed many times:
! (may want to rethink since obs mask is now computed separately per feature)
! ... during each IS_DO_OP, during regridding, and for final LIST, PLOT, etc.
! It would be desirable to avoid this. A way to do this would be
!  1. compute a unique hash based upon the XYZTE context constraints
!  2. save the masks that computed here in lm line memory
!  3. use the hash value and dataset number to recover the saved masks
!  4. clean out the stored masks (say?) when the dataset gets closed

* initialize
	DO i = 1, nobs
	   use_obs(i) = .TRUE.
	ENDDO

* mask at the individual obs level based upon obs cordinates
	DO idim = 1, 4
	   IF (.NOT.cxdsg_constrain (idim))    CYCLE
	   IF (.NOT.cxdsg_has_coord(idim))     CYCLE
	   IF (.NOT.cxdsg_is_obs_coord(idim) ) CYCLE

	   
	   bad = cxdsg_bad_val(idim)
	   lm = cxdsg_coord_lm(idim)

* If the constraint in any direction is just one value, will look for 
* nearest.

	   constrain_lo_lo(idim) = cxdsg_constrain_lo(idim)
	   constrain_hi_hi(idim) = cxdsg_constrain_hi(idim)

	   IF (cxdsg_constrain_lo(idim) .EQ. cxdsg_constrain_hi(idim)) THEN

* Set a new range, the next smallest value than the constraint, and the next largest.
* Or if the coordinates match the constraint, keep that value.e
	      
	      val = ABS(unspecified_val8)
	      del_lo = val
	      del_hi = val
	      DO i = 1, nobs

	         iptr = iobs0+i
	         coord = dsg_linemem(lm)%ptr(iptr)
	         IF (coord .EQ. bad) CYCLE

	         IF (idim .EQ. x_dim) CALL MODSCAT  (
     .		       cxdsg_constrain_lo(idim), 
     .		       cxdsg_constrain_hi(idim), axmod, 1, coord)

	         IF (coord .LE. cxdsg_constrain_lo(idim) .AND. 
     .	             cxdsg_constrain_lo(idim)-coord .LT. del_lo)
     .	            del_lo = cxdsg_constrain_lo(idim) - coord 

		 IF (coord .GT. cxdsg_constrain_hi(idim) .AND. 
     .	             coord-cxdsg_constrain_hi(idim) .LT. del_hi)
     .	            del_hi = coord - cxdsg_constrain_hi(idim)

	      ENDDO

* If the value is at the end of the range of values, only one of these may
* have been set.
	      IF (del_lo.EQ.val .AND. del_hi.EQ.val) THEN
	         del_lo = 0.
	         del_hi = 0.
	     ELSE IF (del_lo .EQ. val) THEN
	         del_lo = del_hi
	      ELSE IF (del_hi .EQ. val) THEN
	         del_hi = del_lo
	      ENDIF

* choose the nearest to the given constraint value 
* TODO: Could return both and then the routine using the data would
*       interpolate if mode_interpolate, etc.

	      IF (del_lo .LE. del_hi) THEN
	         constrain_lo_lo(idim) = cxdsg_constrain_lo(idim) - del_lo
	      ELSE
	         constrain_hi_hi(idim) = cxdsg_constrain_hi(idim) + del_hi
	      ENDIF
	      cx_lo_ww(idim,cx_plot) = cxdsg_constrain_lo(idim) 
	      cx_hi_ww(idim,cx_plot) = cxdsg_constrain_lo(idim) 
	
	   ENDIF ! cxdsg_constrain_lo(idim) .EQ. cxdsg_constrain_hi

	   bad = cxdsg_bad_val(idim)
	   lm = cxdsg_coord_lm(idim)
	   DO i = 1, nobs
	      iptr = iobs0+i
	      coord = dsg_linemem(lm)%ptr(iptr)
	      IF (coord .EQ. bad) CYCLE

	      IF (idim .EQ. x_dim) CALL MODSCAT  (
     .		       constrain_lo_lo(idim), 
     .		       constrain_hi_hi(idim), axmod, 1, coord)

	      use_obs(i) = use_obs(i)
     .		.AND. coord .LE. constrain_hi_hi(idim) 
     .		.AND. coord .GE. constrain_lo_lo(idim)
	   ENDDO
	ENDDO

	RETURN
	END
