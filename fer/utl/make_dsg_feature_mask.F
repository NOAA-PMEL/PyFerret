	SUBROUTINE MAKE_DSG_FEATURE_MASK(dset, cx, fmask, nfeatures )

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
* Create a feature mask for a DSG based upon world coordinate limits
* in the given context

* NOAA/PMEL, Seattle, WA - Tropical Modeling and Analysis Program

* V720 4/17 *sh*
* V73+ *acm* 12/17 INIT_CXDSG needs nfeatures
* V741 6/18 *acm* Mask check enhanced for partial overlap of feature and region
*                 and handle modulo longitudes
* V741 7/18 *acm* Fixes for modulo longitudes in computing whether context
*                 and coordinate regions overlap
* V74+ 8/18 *acm* More on modulo longitudes and the mask
* V76  1/20 *acm* working with Point-type dsg data

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
	INTEGER dset, cx, nfeatures
	LOGICAL fmask(nfeatures)

* internal variable declarations:
        LOGICAL one_or_other, ismod, its_dsg, its_cmpnd, ignore_depth
	INTEGER grid, i, idim, lm, orientation, obsdimlen, feature_line
	REAL    GET_LINE_COORD, coord, ftr_lo(4,nfeatures), ftr_hi(4,nfeatures), 
     .		flo, fhi, axmod, bad, val

	axmod = 360.0

* Will allow depth coordinates to be missing for timeseries or trajectory data

	grid = dsg_xlate_grid(dset)
	CALL TM_DSG_FACTS( grid, orientation, obsdimlen, feature_line, 
     .                     its_dsg, its_cmpnd )
	ignore_depth = orientation.EQ.pfeatureType_Trajectory .OR. 
     .                 orientation.EQ.pfeatureType_Point .OR. 
     .                 orientation.EQ.pfeatureType_TimeSeries

! DSGTBD: presently the same constraints mask gets re-computed many times:
! ... during each IS_DO_OP, during regridding, and for final LIST, PLOT, etc.
! It would be desirable to avoid this. A way to do this would be
!  1. compute a unique hash based upon the XYZTE context constraints
!  2. save the masks that computed here in lm line memory
!  3. use the hash value and dataset number to recover the saved masks
!  4. clean out the stored masks (say?) when the dataset gets closed

* initialize
	cxdsg_constrain_e = .FALSE.
	IF (cx .GT. cx_none) CALL INIT_CXDSG(dset, cx, nfeatures)  ! summarize context ww limits

* preset the features and obs masks based upon E constraints
	IF (cxdsg_constrain_e) THEN
	   DO i = 1, nfeatures
	      fmask(i) = i .GE. cxdsg_lo_e .AND. i .LE. cxdsg_hi_e
	   ENDDO
	ELSE
	   DO i = 1, nfeatures
	      fmask(i) = .TRUE.
	   ENDDO
	ENDIF

* Did the user send in a mask with e.g. "PLOT/FMASK var, maskvals"

	IF (dsg_msk_lm(dset) .NE. unspecified_int4) THEN
	   DO i = 1, nfeatures 
	      val = GET_LINE_COORD (linemem(dsg_msk_lm(dset))%ptr, i)
	      fmask(i) = (val.NE.0. .AND. val.NE.bad_val4) 
	   ENDDO
	ENDIF

* quick exit?
	IF (cxdsg_no_coord_constraints .OR. cx .EQ. cx_none) RETURN

* get the coordinate extremes feature by feature
	CALL DSG_FEATURE_LIMS(dset, nfeatures, ftr_lo, ftr_hi)

* first mask at the feature level based on instance coordinates
	DO idim = 1, t_dim
	   IF (.NOT.cxdsg_constrain(idim)) CYCLE
	   IF (.NOT.cxdsg_has_coord(idim)) CYCLE
   
	   bad = cxdsg_bad_val(idim)
	   IF (cxdsg_is_obs_coord(idim)) THEN

* ... use the coordinate extremes for this feature

	      DO i = 1, nfeatures

	         IF (.NOT.fmask(i)) CYCLE

                 flo = ftr_lo(idim,i)
                 fhi = ftr_hi(idim,i)

		 IF ((flo.EQ.unspecified_val8 .OR. fhi.EQ.unspecified_val8) .AND.
     .		     (idim.EQ.z_dim.OR.idim.EQ.t_dim) .AND. 
     .		      ignore_depth ) CYCLE  ! Depths might be all bad, feature still ok

		 IF (idim .EQ. x_dim) THEN
		    CALL MODSCAT ( cxdsg_constrain_lo(idim), 
     .		       cxdsg_constrain_hi(idim), axmod, 1, flo)
		    CALL MODSCAT ( cxdsg_constrain_lo(idim), 
     .		       cxdsg_constrain_hi(idim), axmod, 1, fhi)
		    ismod = flo.NE.ftr_lo(idim,i) .OR. fhi.NE.ftr_hi(idim,i)
		 ENDIF 

* Does the requested region lie partially or entirely within the feature?
		 one_or_other =
     .              ( (cxdsg_constrain_lo(idim) .GE. flo .AND. 
     .                 cxdsg_constrain_lo(idim) .LE. fhi )      .OR.   
     .		      (cxdsg_constrain_hi(idim) .GE. flo .AND. 
     .		       cxdsg_constrain_hi(idim) .LE. fhi)     )

* or if the feature lies within the requested region that's ok too 
		 IF (.NOT.one_or_other) THEN
		    one_or_other =
     .              ( (flo .GE. cxdsg_constrain_lo(idim) .AND. 
     .                 flo .LE. cxdsg_constrain_hi(idim) )      .OR.   
     .		      (fhi .GE. cxdsg_constrain_lo(idim) .AND. 
     .		       fhi .LE. cxdsg_constrain_hi(idim))     )    
		 ENDIF

		 fmask(i) = fmask(i).AND. one_or_other
	      ENDDO

	   ELSE

* ... use the single point coordinate for this feature

	      lm = cxdsg_coord_lm(idim)
	      DO i = 1, nfeatures
	         coord = dsg_linemem(lm)%ptr(i)
	         IF (coord .EQ. bad) CYCLE
		 IF (idim .EQ. x_dim) CALL MODSCAT  (
     .		       cxdsg_constrain_lo(idim), 
     .		       cxdsg_constrain_hi(idim), axmod, 1, coord)

                 fmask(i) = fmask(i)
     .		  .AND. coord .LE. cxdsg_constrain_hi(idim) 
     .		  .AND. coord .GE. cxdsg_constrain_lo(idim)
	      ENDDO
	   ENDIF
	ENDDO

	RETURN
	END
