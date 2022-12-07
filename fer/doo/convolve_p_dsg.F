	SUBROUTINE CONVOLVE_P_DSG (idim, hlen, wt, dset, nfeatures,
     .				com, com_mr, com_cx,
     .				res, res_mr, res_cx )

*
*
*  This software was developed by the Thermal Modeling and Analysis
*  Project(TMAP) of the National Oceanographic and Atmospheric
*  Administration''s (NOAA) Pacific Marine Environmental Lab(PMEL),
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
* convolve the component grid, com, with the weight function, wt
* along axis idim
* Since this is a DSG the convolution may be only along the 
* "orientation" axis (e.g. T on a Tseries collection) or the
* "instance" axis (always E)
* note: the use of WW constraints on a DSG expression, such as a time range
*       on  a time series collection may result in gaps created in the
*	individual features.  To avoid convolving across these gaps, the
*	observation-level constraints are ignored here. They will be still
*	be used as a mask during commands such as LIST
* also: when bad data points are encountered in the component data all
*	result data depending on it are flagged as bad, too

* programmer - steve hankin
* NOAA/PMEL, Seattle, WA - Tropical Modeling and Analysis Program

* V720 *sh* 5/2017 - based upon COPNVOLVE.F
* V76   1/20 *acm* Orientation of Point-type data set to e_dim
* 10/2022 *acm* Code cleanup: remove comment after # include statement

        IMPLICIT NONE
        include 'tmap_dims.parm'
#	include	"tmap_dset.parm" 
	include	'ferret.parm'
	include	'xcontext.cmn'
	include	'xvariables.cmn'
	include	'xmem_subsc.cmn'
	include	'xdset_info.cmn_text'
	include 'xdyn_linemem.cmn_text'

* calling argument declarations:

	INTEGER	idim, hlen, dset, nfeatures, com_mr, com_cx, res_mr, res_cx
	REAL	wt(-hlen:hlen),
     .          com   ( m1lox:m1hix ),
     .          res   (*)

* internal variable declarations:
	LOGICAL fmask(nfeatures)
	INTEGER	i, ii, base, flen, ifeature, iobs,
     .		orientation, row_size_lm
	REAL	bad_com, bad_res, comp, sum, wsum

* initialize
	bad_com = mr_bad_data( res_mr )
	bad_res = mr_bad_data( com_mr )

* DSG-specific initialize
	orientation = dsg_orientation(dset)
	row_size_lm = dsg_loaded_lm(dsg_row_size_var(dset))

* get feature mask
        CALL MAKE_DSG_FEATURE_MASK(dset, res_cx, fmask, nfeatures)

* Handle z transforms for feature-type of trajectory-profile or timeseries-profile - within this routine
* treat the data type as profile.

	IF (orientation.EQ.pfeatureType_TrajectoryProfile .OR. 
     .	      orientation.EQ.pfeatureType_TimeseriesProfile) orientation = pfeatureType_Profile

* ****** PERFORM THE COMPUTATION *****
	IF (idim .EQ. orientation) THEN
* *** true ragged array processing:
* convolve independently along each feature
*     Work with all data in the feature. WW coordinate masking etc  
*     will be applied beforelisting or plotting

           base = 0    ! obs index at end of preceding feature
* ... loop over the features
	   DO ifeature = 1, nfeatures 

	      flen = dsg_linemem(row_size_lm)%ptr(ifeature)  ! feature length

* ... have the user's constraints removed this feature?
	      IF (.NOT.fmask(ifeature)) THEN
                 base = base + flen
                 CYCLE
              ENDIF

* ... loop over the observations within each feature
              iobs = base
	      DO i = 1, flen
	         iobs = iobs + 1
		 res(iobs) = bad_res

* ... convolve the neighboring points with the weights
	         sum = 0.0
		 wsum = 0.0
	         DO ii = -hlen, hlen

* ... test limits of component data before using
                    IF ( i+ii .GE. 1
     .             .AND. i+ii .LE. flen ) THEN
                       comp = com(iobs+ii)
                    ELSE
	               CYCLE
                    ENDIF

	            IF ( comp .NE. bad_com ) THEN
	               sum = sum + comp*wt(ii)
	               wsum = wsum + wt(ii)
	            ENDIF
	         ENDDO
	         IF (wsum .GT. 0) THEN
	            res(iobs) = sum/wsum
	         ELSE
	            res(iobs) = bad_res
	         ENDIF
	      ENDDO
	      base = base + flen
           ENDDO

	ELSE

* *** a simple list of instances:
* IT IS NOT CLEAR THAT A USE CASE EXISTS for convolving along the instance
* (feature) axis of a DSG.  But we allow it anyway.
* Process all features -- ignoring mask -- for the reasons stated in the intro
	   DO i = 1, nfeatures 
	      sum = 0.0
	      wsum = 0.0
	      res(i) = bad_res

	      DO ii = -hlen, hlen
* ... test limits of component data before using
                 IF ( i+ii .GE. 1
     .          .AND. i+ii .LE. nfeatures ) THEN
                    comp = com(i+ii)
                 ELSE
                    CYCLE
                 ENDIF

	         IF (  comp.NE.bad_com .AND.comp.NE.dsg_masked_out_flag ) THEN
	            sum = sum + comp* wt(ii)
	         ENDIF
	      ENDDO
	      IF (wsum .GT. 0) THEN
	         res(i) = sum/wsum
	      ELSE
	         res(i) = bad_res
	      ENDIF
	   ENDDO
	ENDIF

	RETURN
	END
