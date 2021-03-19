	INTEGER FUNCTION DO_SUM_DSG ( idim, arg, dset, nfeatures, maxobs,
     .				      com, com_mr, com_cx,
     .				      res, res_mr, res_cx,
     .				      counts	)

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
* find the SUM of value of the component grid along the obs or instance 
* axis of a DSG axis

* programmer - Ansley Manke
* NOAA/PMEL, Seattle, WA - SDIG
* 2/2019 from DO_MAX_DSG
* V76   1/20 *acm* Orientation of Point-type data set to e_dim

        IMPLICIT NONE
        include 'tmap_dims.parm'
#	include	"tmap_dset.parm"  ! with feature-type 
	include	'ferret.parm'
	include	'errmsg.parm'
	include	'interp_stack.parm'
	include	'xcontext.cmn'
	include	'xvariables.cmn'
	include	'xmem_subsc.cmn'
	include	'xprog_state.cmn'
	include 'xdyn_linemem.cmn_text'
	include 'xdset_info.cmn_text'
	include 'xdsg_context.cmn'

* calling argument declarations:
	INTEGER	idim, com_mr, com_cx, res_mr, res_cx, dset, nfeatures,
     .		maxobs
	REAL	arg, 
     .          com   (*),
     .          res   (*),  ! E=@AVE will have -999:-999
     .          counts(*)

* internal variable declarations:
	LOGICAL fmask(nfeatures), omask(maxobs)
	INTEGER ok, i, ifeature, iobs, flen, base, orientation, row_size_lm
	REAL	bad_res, bad_com, val

* internal parameter declaration
	REAL		too_small
	PARAMETER     ( too_small = -1.E33 )

* initialize
	orientation = dsg_orientation(dset)

	IF (dsg_row_size_var(dset) .GT. 0) row_size_lm = dsg_loaded_lm(dsg_row_size_var(dset))
	bad_res = mr_bad_data( res_mr )
	bad_com = mr_bad_data( com_mr )

* get feature mask
        CALL MAKE_DSG_FEATURE_MASK(dset, com_cx, fmask, nfeatures)

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

	      counts(ifeature) = 0.
	      res(ifeature) = 0.

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

* ... the input data value
                 val = com(iobs)

	         IF ( val .EQ. bad_com ) CYCLE
	         counts(ifeature) = counts(ifeature) + 1.
	         res(ifeature) = res(ifeature) + val
              ENDDO

	      base = base + flen

           ENDDO

	ELSE

* *** a simple list of instances:
* compress along the instance (feature) axis resulting in just a single value
	   res(1) = 0      ! initialize sum
	   counts(1) = 0   ! number of valid component values found
	   DO ifeature = 1, nfeatures 

* ... have the user's WW constraints removed this feature?
	      IF (.NOT.fmask(ifeature)) CYCLE

	      IF ( com(ifeature) .EQ. bad_com ) CYCLE
	      counts(1) = counts(ifeature) + 1.
	      res(1) = res(1) + com(ifeature)
           ENDDO

	ENDIF

	RETURN
	END
