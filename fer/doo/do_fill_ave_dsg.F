	INTEGER FUNCTION DO_FILL_AVE_DSG ( idim, haf, dset, nfeatures, maxobs,
     .				com, com_mr, com_cx,
     .				res, res_mr, res_cx)

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
* fill along axis idim by a running mean window
* for data on Distributed Sampling Geometries grids
*
* programmer - Ansley Manke
* NOAA/PMEL, Seattle, WA - Science Data Integration Group
* 2/2019
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
	INTEGER	idim, haf, dset, nfeatures, maxobs, 
     .          com_mr, com_cx, res_mr, res_cx
	REAL	com   (*),
     .          res   (*)

* internal variable declarations:
	LOGICAL	fmask(nfeatures), omask(maxobs)
	INTEGER	i, ii,ii_com, ifeature, iobs, flen, base, 
     .		orientation, row_size_lm, dsg_fmt_grid, dsg_coord_lm, i1
	REAL	bad_res, bad_com, sum, cnt, val

* initialize
	bad_res = mr_bad_data( com_mr )

* initialize
	orientation = dsg_orientation(dset)

	IF (dsg_row_size_var(dset) .GT. 0) row_size_lm = dsg_loaded_lm(dsg_row_size_var(dset))
	bad_res = mr_bad_data( res_mr )
	bad_com = mr_bad_data( com_mr )

* get feature mask and dsg grid and coordinate information
        CALL MAKE_DSG_FEATURE_MASK(dset, res_cx, fmask, nfeatures)

	row_size_lm  = dsg_loaded_lm(dsg_row_size_var(dset))
	dsg_fmt_grid = dsg_xlate_grid(dset)
	orientation  = dsg_orientation(dset)
	dsg_coord_lm = cxdsg_coord_lm(orientation)

* Handle z transforms for feature-type of trajectory-profile or timeseries-profile - within this routine
* treat the data type as profile.

	IF (orientation.EQ.pfeatureType_TrajectoryProfile .OR. 
     .	      orientation.EQ.pfeatureType_TimeseriesProfile) orientation = pfeatureType_Profile

* ****** PERFORM THE COMPUTATION *****
	IF (idim .EQ. orientation) THEN

* *** true ragged array processing:

	   base = 0    ! obs index at end of preceding feature

* ... loop over the features
	   ftrloop: DO ifeature = 1, nfeatures 

	      flen = dsg_linemem(row_size_lm)%ptr(ifeature)  ! feature length
              iobs = base

* ... have the user's WW constraints removed this feature?
	      IF (.NOT.fmask(ifeature)) THEN
	         DO i = 1, flen
	            iobs = iobs + 1   ! index in the contig ragged array
	            res(iobs) = bad_res
		 ENDDO
                 base = base + flen
                 CYCLE  ftrloop
              ENDIF

* ... get observation-level mask for this feature
	      CALL MAKE_DSG_OBS_MASK(dset, com_cx, ifeature, base,
     .				     omask, flen)

* ... loop over the observations within each feature
	      oloop: DO i = 1, flen
	         iobs = iobs + 1   ! index in the contig ragged array
	         res(iobs) = bad_res
	         IF (.NOT.omask(i)) CYCLE oloop

* ... the input data value

                 val = com(iobs)

	         IF ( val .NE. bad_com ) THEN

		    res(iobs) = val

	         ELSE
		 
	            cnt = 0
	            sum = 0.0
	            DO ii = -haf, haf
	               ii_com = i + ii + base
	               IF ( ii_com.LE.base .OR. ii_com.GT.flen+base ) CYCLE
	               val = com(ii_com)
	               IF ( val .NE. bad_res ) THEN
	                  cnt = cnt + 1
	                  sum = sum + val
	               ENDIF
	            ENDDO

	            IF ( cnt .GT. 0 ) THEN
	               res(iobs) = sum / cnt
	            ELSE
	              res(iobs) = bad_res
	            ENDIF

	         ENDIF

              ENDDO  oloop

              base = base + flen

	   ENDDO  ftrloop

	ELSE

* *** a simple list of instances:

	   DO ifeature = 1, nfeatures 

* ... have the user's WW constraints removed this feature?
	      
	      IF (.NOT.fmask(ifeature)) THEN
	         res(ifeature) = bad_res
		 CYCLE
	      ENDIF

* ... the input data in the nfeature direction
              val = com(ifeature)

	      IF ( val .NE. bad_com ) THEN
	         res(ifeature) = val
	      ELSE
	      
	         cnt = 0
	         sum = 0.0
	         DO ii = -haf, haf
	            ii_com = ifeature + ii
	            IF ( ii_com.LT.1 .OR. ii_com.GT.nfeatures ) CYCLE
	            IF ( .NOT.fmask(ii_com) ) CYCLE
	            val = com(ii_com)
	            IF ( val .NE. bad_res ) THEN
	               cnt = cnt + 1
	               sum = sum + val
	            ENDIF
	         ENDDO

	         IF ( cnt .GT. 0 ) THEN
	            res(ifeature) = sum / cnt
	         ELSE
	            res(ifeature) = bad_res
	         ENDIF

	      ENDIF

           ENDDO  ! ifeature

	ENDIF     ! simple list of instances


* always success
	DO_FILL_AVE_DSG = ferr_ok
	RETURN

	END

