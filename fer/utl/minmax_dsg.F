	SUBROUTINE MINMAX_DSG ( dset, cx, nfeatures, idim, array,  
     .        bad_flag, lo, hi, nok )

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
* determine the minimum and maximum values within a REAL array
* for data on Discrete Sampling Geometries grid. Return the overall 
* min and max of the data after feature-masking

* programmer - Ansley Manke
* NOAA/PMEL, Seattle, WA - Science Data Integration Group
* 3/2019

        IMPLICIT NONE
	include	'ferret.parm'
        include 'tmap_dims.parm'
	include 'xdyn_linemem.cmn_text'
	include 'xdset_info.cmn_text'
	include 'xdsg_context.cmn'

* calling argument declarations:
	INTEGER	dset, cx, nfeatures, idim, n, nok
	REAL    array(*), bad_flag, lo, hi

* internal variable declarations:
	LOGICAL fmask(nfeatures)
	INTEGER i, ifeature, iobs, flen, base, 
     .          orientation, row_size_lm

* DSG-specific initialize

        CALL MAKE_DSG_FEATURE_MASK(dset, cx, fmask, nfeatures)
	row_size_lm  = dsg_loaded_lm(dsg_row_size_var(dset))
	orientation  = dsg_orientation(dset)

* initialize
	lo = arbitrary_large_val4
	hi = arbitrary_small_val4
	nok = 0

	IF (idim .EQ. orientation) THEN 

* *** true ragged array processing:
*     return the overall min and max of the data after feature-masking

           base = 0    ! obs index at end of preceding feature

* ... loop over the features
	   ftrloop: DO ifeature = 1, nfeatures 

	      flen = dsg_linemem(row_size_lm)%ptr(ifeature)  ! feature length

* ... have the user's constraints removed this feature?
	      IF (.NOT.fmask(ifeature)) THEN
                 base = base + flen
                 CYCLE ftrloop
              ENDIF

* ... loop over the observations within each feature
              iobs = base

	      iobs = base
	      DO i = 1, flen
	         iobs = iobs + 1   ! index in the contig ragged array

	         IF ( array(iobs) .EQ. bad_flag ) CYCLE
	         IF ( array(iobs) .LT. lo ) lo = array(iobs)
	         IF ( array(iobs) .GT. hi ) hi = array(iobs)
	         nok = nok + 1
	      ENDDO

	      base = base + flen

	   ENDDO ftrloop

	ELSE  

* *** a simple list of instances:
* Work along the feature-number axis

	   DO ifeature = 1, nfeatures 

* ... have the user's constraints removed this feature?
	      IF (.NOT.fmask(ifeature)) CYCLE

	      IF ( array(ifeature) .EQ. bad_flag ) CYCLE
	      IF ( array(ifeature) .LT. lo ) lo = array(ifeature)
	      IF ( array(ifeature) .GT. hi ) hi = array(ifeature)
	      nok = nok + 1
           ENDDO


	ENDIF  ! obs vs feature
	
	RETURN
	END
