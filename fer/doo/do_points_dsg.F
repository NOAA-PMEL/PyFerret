	SUBROUTINE DO_POINTS_DSG(	action, idim, 
     .				dset, nfeatures, maxobs,
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
* determine the number of good and number of bad points along the given axis

* programmer - steve hankin
* NOAA/PMEL, Seattle, WA - Tropical Modeling and Analysis Program
* written for VAX computer under VMS operating system
*
* V230:   2/3/92 - based on DO_AVE_INT
*        2/12 *kms* Add E and F dimensions (use nferdims in tmap_dims.parm)
* V71 *acm* 12/16 this routine now performs accumulation, possibly continuing
*                 over several chunks, so keep adding to the count in res(). 
*                 Initialization and finalizing called separately from IS_TRANS
* V76   1/20 *acm* Orientation of Point-type data set to e_dim
*           1/21 *acm* Implement transform in z direction of trajectoryprofile or timeseriesprofile data.
*                                   Note that the f-mask includes any subseting done for world coordinates, as well
*                                   as any mask defined for profiles  (Defining a mask with SET DATA/FMASK=mask1_var )
*                                   or for selecting trajectory or timeseries (Defining a mask with SET DATA/SMASK=mask2_var)

        IMPLICIT NONE
        include 'tmap_dims.parm'
#	include	"tmap_dset.parm"  ! with feature-type 
	include	'ferret.parm'
	include	'interp_stack.parm'
	include	'xcontext.cmn'
	include	'xvariables.cmn'
	include	'xmem_subsc.cmn'
	include 'xunits.cmn_text'
	external xunits_data
	include 'xtm_grid.cmn_text'
	external xgt_grid_data
	include 'xdyn_linemem.cmn_text'
	include 'xdset_info.cmn_text'

* calling argument declarations:
	INTEGER	action, idim, dset, nfeatures, maxobs, com_mr, com_cx, res_mr, res_cx
	REAL    com   (*),
     .          res   (*)  ! E=@AVE will have -999:-999

* internal variable declarations:
	LOGICAL	good_un, fmask(nfeatures), omask(maxobs)
	INTEGER i, pt, flen, base, ifeature, iobs, ok, cnt,
     .          orientation, row_size_lm
	REAL	bad_com, bad_res, val


* --- end of introductory code ---


* initialize
        pt   = cx_lo_ss( res_cx, idim )
	bad_com = mr_bad_data( com_mr )
        bad_res = mr_bad_data( res_mr )
	good_un  = action .EQ. isact_good_pt

	orientation = dsg_orientation(dset)

	IF (dsg_row_size_var(dset) .GT. 0) row_size_lm = dsg_loaded_lm(dsg_row_size_var(dset))
	bad_res     = mr_bad_data( res_mr )

* get feature mask
        CALL MAKE_DSG_FEATURE_MASK(dset, res_cx, fmask, nfeatures)

* Handle z transforms for feature-type of trajectory-profile or timeseries-profile - within this routine
* treat the data type as profile.

	IF (orientation.EQ.pfeatureType_TrajectoryProfile .OR. 
     .	      orientation.EQ.pfeatureType_TimeseriesProfile) orientation = pfeatureType_Profile

* ****** PERFORM THE COMPUTATION *****
	IF (idim .EQ. orientation) THEN

* *** true ragged array processing:
* compress each feature into a single value 
           base = 0    ! obs index at end of preceding feature
* ... loop over the features
	   DO ifeature = 1, nfeatures 

	      res(ifeature) = bad_res ! initialize

	      ok = 0   ! number of valid component values found
	      cnt = 0  ! count points not masked

	      flen = dsg_linemem(row_size_lm)%ptr(ifeature)  ! feature length

* ... have the user's WW constraints removed this feature?
	      IF (.NOT.fmask(ifeature)) THEN
                 base = base + flen
                 CYCLE
              ENDIF

* ... get observation-level mask for this feature
	      CALL MAKE_DSG_OBS_MASK(dset, com_cx, ifeature, base,
     .				     omask, flen)
	
* ... loop over the observations within each feature
              iobs = base
	      DO i = 1, flen
	         iobs = iobs + 1   ! index in the contig ragged array

	         IF (.NOT.omask(i)) CYCLE
		 cnt = cnt + 1	   

* ... the input data value
                 val = com(iobs)

	         IF ( val .EQ. bad_com ) CYCLE
	         ok = ok + 1.
              ENDDO

	      IF (good_un) THEN
	         res(ifeature) = ok
              ELSE
                 res(ifeature) = cnt - ok
              ENDIF 
	      
	      base = base + flen
           ENDDO

	ELSE

* *** a simple list of instances:
* look along the instance (feature) axis resulting in just a single value

	   ok = 0   ! number of valid component values found
	   cnt = 0  ! count points not masked
	   DO ifeature = 1, nfeatures 

* ... have the user's WW constraints removed this feature?
	      IF (.NOT.fmask(ifeature)) CYCLE
	      cnt = cnt + 1

* ... the input data value
              val = com(ifeature)

	      IF ( val .EQ. bad_res ) CYCLE
	      ok = ok + 1.
           ENDDO
	   
	   IF (good_un) THEN
	      res(1) = ok
	   ELSE
              res(1) = cnt - ok
	   ENDIF 

	ENDIF

	RETURN
	END