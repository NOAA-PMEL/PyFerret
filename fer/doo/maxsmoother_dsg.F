	SUBROUTINE MAXSMOOTHER_DSG( idim, wlen, dset, nfeatures,
     .				com, com_mr, com_cx,
     .				res, res_mr, res_cx, 
     .				list )

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
* Compute the minimum of the requested length, along axis idim
* for data on Distributed Sampling Geometries grids
* note: the component context may not be of adequate size for the full
*	calculation.  Missing data flags will be inserted where computation is
*	impossible. If there is any data present in the window, the minimum
*       is computed based on that data. Minima are computed with the data
*       that is available in partial windows at the edges of the domain. 

* Ansley Manke
* NOAA/PMEL, Seattle, WA - Science Data Integration Group
*  3/2019
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
	INTEGER	idim, wlen, dset, nfeatures,
     .		com_mr, com_cx, res_mr, res_cx
	REAL	com   ( * ),
     .          res   ( * ), 
     .          list(*)

* internal variable declarations:
	LOGICAL fmask(nfeatures)
	INTEGER	i, ii, ndx, lo_lim, hi_lim, hlen, ifeature, iobs,
     .          flen, base, orientation, row_size_lm
	REAL    bad_com, bad_res, comp, cmax

* initialize
	bad_com = mr_bad_data( res_mr )
	bad_res = mr_bad_data( com_mr )

	hlen = wlen/2
	IF ( MOD(wlen,2) .EQ. 0)  hlen = (wlen+1)/2
	
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
	      
              lo_lim = 1
              hi_lim = flen

* ... loop over the observations within each feature
              iobs = base
	      DO i = 1, flen
	         iobs = iobs + 1

* ... convolve the neighboring points with the weights

	         ndx = 0
		 cmax = -1. * ABS(bad_com)
	         DO ii = -hlen, hlen
* ... test limits of component data before using
                    IF ( i+ii .GE. lo_lim
     .             .AND. i+ii .LE. hi_lim ) THEN
                       comp = com(iobs+ii)
                    ELSE
                       comp = bad_com
                    ENDIF

                    IF ( comp .NE. bad_com ) THEN
                       cmax = MAX(cmax, comp)
                       ndx = ndx + 1
                    ENDIF
	         ENDDO

	         IF (ndx .EQ. 0) THEN 
                    res(iobs) = bad_res
	         ELSE
                    res(iobs) = cmax
	         ENDIF

	      ENDDO
	      base = base + flen
	   ENDDO

	ELSE

* *** a simple list of instances:
* Process all features -- ignoring mask -- for the reasons stated in the intro

	   lo_lim = 1
	   hi_lim = nfeatures

	   DO ifeature = 1, nfeatures 

	      ndx = 0
	      DO ii = -hlen, hlen
* ... test limits of component data before using
                 IF ( ifeature+ii .GE. lo_lim
     .          .AND. ifeature+ii .LE. hi_lim ) THEN
                    comp = com(ifeature+ii)
                 ELSE
                    comp = bad_res
                 ENDIF
		 
                 IF ( comp.NE.bad_com .AND.comp.NE.dsg_masked_out_flag) THEN
                    cmax = MAX(cmax, comp)
                    ndx = ndx + 1
                 ENDIF

	         IF (ndx .EQ. 0) THEN 
                   res(ifeature) = bad_res
	         ELSE
                   res(ifeature) = cmax
	         ENDIF

	      ENDDO

	   ENDDO

	ENDIF

	RETURN
	END
