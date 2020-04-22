	SUBROUTINE DSG_OBS_MARK_GAPS (dset, cx, nfeatures, nobs_total, vals)

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
* programmer - Ansley Manke
* NOAA/PMEL, Seattle, WA - SDIG program
* 4/24/2018
*
* Define a variable of length nobs_total where the end of each feature 
* is marked by the value 1. The routine plotxy_ribbon can then treat 
* this as agap-marking variable and pick up the pen after drawing to the 
* associated points.

* programmer - Ansley Manke  
* NOAA/PMEL, Seattle, WA - Science Data Integration Group
*
* V7.4+ *acm*  4/2018
* V7.5+ *acm*  6/2019  Do obs masking as we go, so if using just subsets of features 
*                      the gaps are in the right places2018
* V7.51 *acm*  7/2019  If the length sent in is the nfeatures length, send back an
*                      array that's all zero.

	include 'tmap_dims.parm'
	include 'xdset_info.cmn_text'
	include 'xdyn_linemem.cmn_text'

* calling argument declarations:
	INTEGER	dset, cx, nfeatures, nobs_total
	REAl vals(*)


* internal variable declarations:
	LOGICAL process_feature(nfeatures), process_obs(nobs_total)
	INTEGER	ifeature, iobs, nobs, row_size_lm, base, fobs, nmasked, nobsf

	row_size_lm = dsg_loaded_lm(dsg_row_size_var(dset))

* If the operation's being done on a feature-variable, set the "gaps" to all 0.
* e.g. a plot of lon vs lat for a set of timeseries stations.

	IF (nobs_total .EQ. nfeatures) THEN
	   DO iobs = 1, nobs_total
	      vals(iobs) = 0.
	   ENDDO

	   RETURN

	ENDIF

* create variable that's 1 at the start of each trajectory, 0 otherwise

	CALL MAKE_DSG_FEATURE_MASK(dset, cx, process_feature, nfeatures)


	nmasked = 0  ! overall count after masking
	base = 0     ! obs index at end of preceding feature
	fobs = 0     ! count of masked data in each feature
	
	DO ifeature = 1, nfeatures
	   nobsf = dsg_linemem(row_size_lm)%ptr(ifeature)  ! feature length
	   IF (process_feature(ifeature)) THEN

* ... define observation-level mask for this feature
	      CALL MAKE_DSG_OBS_MASK(dset, cx, ifeature, base, process_obs, nobsf)
	      fobs = 0
	      DO iobs = 1, nobsf
	         IF (process_obs(iobs)) THEN
		    fobs = fobs+1
		    nmasked = nmasked+1
		    vals(nmasked) = 0.
	         ENDIF

* ...  if masking keeps some but not all elements of feature, mark a gap.
	         IF (iobs.GT.1 .AND. process_obs(iobs) .AND. nmasked.GT.1) THEN
		    IF (.NOT.process_obs(iobs-1) ) vals(nmasked-1) = 1.
		 ENDIF
	      ENDDO

* Mark last element of feature, if it had any valid data

	      IF (fobs.GT.0) vals(nmasked) = 1.

	   ENDIF
	   
	   base = base + nobsf
	ENDDO

* End of last feature isn't a gap.
	IF (nmasked.GT.0) vals(nmasked) = 0.  

	RETURN
	END

