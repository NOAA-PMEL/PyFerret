	SUBROUTINE PLOT_DSG_APPLY_OBS_MASK (dset, cx, nfeatures, 
     .       nobs_total, vals, nmasked)

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
* Apply feature and obs mask from a DSG dataset to an array

* programmer - Ansley Manke
* NOAA/PMEL, Seattle, WA - SDIG program
* 4/24/2018

	include 'tmap_dims.parm'
	include 'xdset_info.cmn_text'
	include 'xdyn_linemem.cmn_text'

* calling argument declarations:
	INTEGER	dset, cx, nfeatures, nobs_total, nmasked
	REAl vals(*)

* internal variable declarations:
	LOGICAL process_feature(nfeatures), process_obs(nobs_total)
	INTEGER	irow_l, irow_f, ifeature, iobs, base, nobs, row_size_lm

	row_size_lm = dsg_loaded_lm(dsg_row_size_var(dset))

* create a mask showing which features to plot

	DO iobs = 1, nobs_total
	   process_obs(iobs) = .FALSE.
	ENDDO

	CALL MAKE_DSG_FEATURE_MASK(dset, cx, process_feature, nfeatures)
	         

	CALL MASK_DSG_OBSVAR (nfeatures, process_feature, 
     .        process_obs, dsg_linemem(row_size_lm)%ptr, dset, cx, 
     .        vals, vals, nmasked)


	RETURN
	END

