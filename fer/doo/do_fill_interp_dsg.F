	INTEGER FUNCTION DO_FILL_INTERP_DSG( idim, dset, nfeatures, maxobs,
     .				com, com_mr, com_cx,
     .				res, res_mr, res_cx,
     .				coords )


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
* fill along axis idim by interpolating from the closest available valid pt
* for data on Distributed Sampling Geometries grids


* programmer - Ansley Manke
* NOAA/PMEL, Seattle, WA - Science Data Integration Group
* 2/2019
* V76   1/20 *acm* Orientation of Point-type data set to e_dim
* 10/2022 *acm* Code cleanup: remove comment after # include statement

        IMPLICIT NONE
	include 'tmap_dims.parm'
	include 'xtm_grid.cmn_text'
	external xgt_grid_data
#	include	"tmap_dset.parm" 
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
	INTEGER	idim, com_mr, com_cx, res_mr, res_cx, dset, nfeatures, maxobs
	REAL	coords(*)
	REAL	com   (*),
     .          res   (*)

* internal variable declarations:
	INTEGER i, ii, lo_index, hi_index, look, i1
	REAL	bad_res, bad_com, lo_coord, hi_coord, coord, val, 
     .          lo_val, hi_val, boxdel, boxmean, boxstd

	LOGICAL	fmask(nfeatures), omask(maxobs), no_lo_val
	INTEGER ifeature, iobs, flen, base, orientation, row_size_lm, 
     .          dsg_fmt_grid, dsg_coord_lm

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

* ... For coordinates use index values within the feature. 
*     DSGTODO: Use coordinate data such as time or depth, 
*     and earth-distance? If so compute per feature

	DO i = 1, maxobs
	   coords(i) = i
	ENDDO
	   
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

* Get and save box sizes for all feature coordinates. Feature-mask applied later.
* For coordinates, distance-along-path, for trajectories, or the coordinate data 
* for time or profile types

              IF (orientation .EQ. x_dim) THEN

	         val = 1.  ! the unit, not applied here
	         CALL BOXES_DSG (orientation, flen, val, 
     .		   dsg_linemem(dsg_coord_lm)%ptr(base+1), 
     .		   dsg_linemem(cxdsg_coord_lm(y_dim))%ptr(base+1), coords)
           
* array coords congains the box sizes.
* Sum the box-sizes along the trajectory to create the "coordinate" list
* Large box-sizes greater than 2 std are replaced by nearby box size

	         coords(1) = 0.
	         DO i = 2, flen

		    boxdel = coords(i)

		    IF (boxdel .GT. 2.* boxstd) THEN
		       IF (i.GT.1) THEN
		          boxdel = MIN( boxstd, coords(i-1) )
		       ELSEIF (i.LT.flen) THEN
		          boxdel = MIN( boxstd, coords(i+1) )
		       ELSE
		          boxdel = boxmean
		       ENDIF
	            ENDIF

	            coords(i) = coords(i-1) + boxdel
	         ENDDO



              ELSE
	
* The coordinates for this feature.
	         iobs = base
	         DO i = 1, flen
	            iobs = iobs + 1
                    coords(i) = dsg_linemem(dsg_coord_lm)%ptr(iobs)
		 ENDDO

              ENDIF


	      iobs = base

* ... get observation-level mask for this feature
	      CALL MAKE_DSG_OBS_MASK(dset, com_cx, ifeature, base,
     .				     omask, flen)

* ... Initialize - is the first data in this feature valid?

              no_lo_val = .TRUE.
              val = com(iobs+1 )
	      IF (val .NE. bad_com ) THEN
                 lo_val = val
	         lo_index = 1
	         lo_coord = coords(1)
                 no_lo_val = .FALSE.
	      ENDIF

* ... loop over the observations within each feature
	      oloop: DO i = 1, flen
	         iobs = iobs + 1   ! index in the contig ragged array
	         res(iobs) = bad_res

	         IF (.NOT.omask(i)) CYCLE oloop

* ... the input data value
                 val = com(iobs)

	         IF ( val .NE. bad_com ) THEN

		    res(iobs) = val
		    lo_val = val
		    lo_index = i
		    lo_coord = coords(i)
		    no_lo_val = .FALSE.

	         ELSE
		 
		    IF (no_lo_val) THEN
* ... no lower valid point
		       res(iobs) = bad_res
		       CYCLE oloop
		    ENDIF

* ... look forward from current position for next valid point

                    hi_index = 0
		    lookup: DO ii = i+1, flen
                       look = ii
		       IF ( com(ii+base) .NE. bad_com ) THEN
                          hi_val = com(ii+base)
	                  hi_index = ii
		          hi_coord = coords(ii)
			  EXIT  lookup
                       ENDIF
                    ENDDO lookup

		    IF (hi_index .NE. 0) THEN

* ... fill in interpolating value
		       coord    = coords(i)
		       res(iobs) = ( hi_val*(coord-lo_coord)
     .                            +  lo_val*(hi_coord-coord) )
     .                            / (hi_coord - lo_coord)
		    ENDIF

	         ENDIF

              ENDDO  oloop

              base = base + flen

	   ENDDO  ftrloop



	ELSE

* *** a simple list of instances:


	   DO ifeature = 1, nfeatures
	      coords(ifeature) = ifeature
	      res(ifeature) = bad_res
	   ENDDO

	   no_lo_val = .TRUE.
	   DO i1 = 1, nfeatures
	      IF (fmask(i1) ) EXIT
	   ENDDO
	   val = com(i1 )
	   
	   IF (val .NE. bad_com) THEN
	      lo_val = val
	      lo_index = i1
	      lo_coord = coords(i1)
	      no_lo_val = .FALSE.
	   ENDIF

	   DO ifeature = i1, nfeatures 

* ... have the user's WW constraints removed this feature?
	      IF (.NOT.fmask(ifeature)) CYCLE

* ... the input data value
              val = com(ifeature)

	      IF ( val .NE. bad_com ) THEN
	         res(ifeature) = val
		 lo_val = val
		 lo_index = ifeature
		 lo_coord = coords(ifeature)
		 no_lo_val = .FALSE.

	      ELSE

	         IF (no_lo_val) THEN
* ... no lower valid point
	            res(iobs) = bad_res
	            CYCLE
		 ENDIF

* ... look forward from current position for next valid point

		 hi_index = 0
		 DO ii = ifeature+1, nfeatures
		    look = ii
		    IF (.NOT.fmask(ii)) CYCLE
		    IF ( com(ii) .NE. bad_com ) THEN
                       hi_val = com(ii)
	               hi_index = look
		       hi_coord = coords(ii)
	               CYCLE  ! this loop
		    ENDIF
		 ENDDO

		 IF (hi_index .NE. 0) THEN

* ... fill in interpolating value
		    coord    = coords(i)
		    res(ifeature) = ( hi_val*(coord-lo_coord)
     .                             +  lo_val*(hi_coord-coord) )
     .                            / (hi_coord - lo_coord)
		 ENDIF

	      ENDIF

           ENDDO  ! ifeature

	ENDIF     ! simple list of instances


* always success
	DO_FILL_INTERP_DSG = ferr_ok
	RETURN

	END
