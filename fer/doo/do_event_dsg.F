	SUBROUTINE DO_EVENT_DSG( idim, arg, dset, nfeatures, maxobs,
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
* the indicated argument value is regarded as an "event threshhold".
* Any time that the component variable crosses or lands on this value
* an "event" occurs and the output should step up by "1" to indicate it.
* If an exact match occurs, step up ON that value.
* If a crossing occurs, step up on the first value AFTER the crossing.  
*  * for data on Distributed Sampling Geometries grids - operate on each feature


* Ansley Manke
* NOAA/PMEL, Seattle, WA - Science Data Integration Group
*  3/2019
* V76   1/20 *acm* Orientation of Point-type data set to e_dim

        IMPLICIT NONE
        include 'tmap_dims.parm'
#	include	"tmap_dset.parm"  ! with feature-type 
	include	'ferret.parm'
	include	'xcontext.cmn'
	include	'xvariables.cmn'
	include	'xmem_subsc.cmn'
	include	'xdset_info.cmn_text'
	include 'xdyn_linemem.cmn_text'

* calling argument declarations:
	INTEGER	idim, wlen, dset, nfeatures, maxobs,
     .		com_mr, com_cx, res_mr, res_cx
	REAL	arg, 
     .          com   ( * ),
     .          res   ( * )

* internal variable declarations:
	LOGICAL fmask(nfeatures), omask(maxobs), first_pt, first_lt 
	INTEGER	i, ifeature, iobs,flen, base, orientation, 
     .          row_size_lm, step_val
	REAL    bad_com, bad_res, comp


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
* find events along each feature
* skip data masked out by WW masking on the obs axis

*** currently this is the max-smoother: needs event logic

	   base = 0    ! obs index at end of preceding feature
* ... loop over the features
	   DO ifeature = 1, nfeatures 

	      flen = dsg_linemem(row_size_lm)%ptr(ifeature)  ! feature length

* ... have the user's constraints removed this feature?

	      IF (.NOT.fmask(ifeature)) THEN
                 base = base + flen
                 CYCLE
              ENDIF

* ... get observation-level mask for this feature
	      CALL MAKE_DSG_OBS_MASK(dset, com_cx, ifeature, base,
     .				     omask, flen)

* ... loop over the observations within each feature

              first_pt = .TRUE.
	      step_val = 0

              iobs = base
	      DO i = 1, flen
	         iobs = iobs + 1
		 
	         IF (.NOT.omask(i)) CYCLE

	         comp = com(iobs)
                 IF ( comp .EQ. bad_com ) THEN
                    first_pt = .TRUE.
	         ELSEIF (comp .EQ. arg) THEN
	            step_val = step_val + 1
	            first_pt = .TRUE.
		 ELSEIF ( first_pt ) THEN
* ... first non-bad value
                    first_pt = .FALSE.
                    first_lt = comp .LT. arg
                 ELSEIF ( first_lt .NEQV. (comp .LT. arg) ) THEN
* ... found a "crossing"
	            step_val = step_val + 1
	            first_lt = .NOT. first_lt
                 ENDIF

	         res(iobs) = DBLE(step_val)

	      ENDDO
	      base = base + flen
	   ENDDO

	ELSE

* *** a simple list of instances:
* Process all features -- ignoring mask -- for the reasons stated in the intro

	   DO ifeature = 1, nfeatures 

	      comp = com(ifeature)
	      IF ( comp.EQ.bad_com .OR. comp.EQ.dsg_masked_out_flag) THEN
	         first_pt = .TRUE.
	      ELSEIF (comp .EQ. arg) THEN
	         step_val = step_val + 1
	         first_pt = .TRUE.
	      ELSEIF ( first_pt ) THEN
* ... first non-bad value
	         first_pt = .FALSE.
	         first_lt = comp .LT. arg
	      ELSEIF ( first_lt .NEQV. (comp .LT. arg) ) THEN
* ... found a "crossing"
	         step_val = step_val + 1
	         first_lt = .NOT. first_lt
	      ENDIF

	      res(ifeature) = DBLE(step_val)

	   ENDDO

	ENDIF

	RETURN
	END
