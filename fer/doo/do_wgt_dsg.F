	SUBROUTINE DO_WGT_DSG( action, idim, dset, nfeatures, 
     .				com, com_mr, com_cx,
     .				res, res_mr, res_cx,
     .				box )

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
* Return the weights as they would be used for an average or integral
* for data on Distributed Sampling Geometries grids

* programmer - Ansley Manke
* NOAA/PMEL, Seattle, WA - Science Data Integration Group
* 2/2019
* V76   1/20 *acm* Orientation of Point-type data set to e_dim
* 10/2022 *acm* Code cleanup: remove comment after # include statement

        IMPLICIT NONE
        include 'tmap_dims.parm'
#	include	"tmap_dset.parm" 
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
	include 'xdsg_context.cmn'

* calling argument declarations:
	INTEGER	action, idim, dset, com_mr, com_cx, res_mr, res_cx, 
     .		nfeatures
	REAL	com(*),
     .          res(*), 
     .		box(*)

* internal variable declarations:
	LOGICAL fmask(nfeatures)
	INTEGER i, grid, iaxis, ok, ifeature, iobs, 
     .          flen, base, orientation, row_size_lm, 
     .          dsg_fmt_grid, dsg_coord_lm, dsg_coord_lm_y
	REAL	BOX_SIZE, unit, bad_com, bad_res, 
     .          dummy, boxdel, boxmean, boxstd
	CHARACTER*80 errstr

* --- end of introductory code ---

* initialize
	grid = mr_grid( com_mr )
	bad_com = mr_bad_data( com_mr )
	bad_res = mr_bad_data( res_mr )

* DSG-specific initialize

        CALL MAKE_DSG_FEATURE_MASK(dset, com_cx, fmask, nfeatures)
	row_size_lm  = dsg_loaded_lm(dsg_row_size_var(dset))
	dsg_fmt_grid = dsg_xlate_grid(dset)
	orientation  = dsg_orientation(dset)
	dsg_coord_lm = cxdsg_coord_lm(orientation)
	IF (orientation .EQ. x_dim) dsg_coord_lm_y = cxdsg_coord_lm(y_dim)
	
* perform integrations on standard units if possible
	unit = 1
	iaxis = grid_line(orientation, dsg_fmt_grid)
	unit = un_convert( line_unit_code(iaxis) )	   

* Handle z transforms for feature-type of trajectory-profile or timeseries-profile - within this routine
* treat the data type as profile.

	IF (orientation.EQ.pfeatureType_TrajectoryProfile .OR. 
     .	      orientation.EQ.pfeatureType_TimeseriesProfile) orientation = pfeatureType_Profile

* ****** PERFORM THE COMPUTATION *****
	IF (idim .EQ. orientation) THEN

* *** true ragged array processing:
* work along each feature 
*     Work with all data in the feature. WW coordinate masking etc  
*     will be applied beforelisting or plotting

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

* Single item in the feature, the average is just the source value; the integral
* can't be computed

	      IF (flen .EQ. 1) THEN 
	          iobs = iobs + 1 
		  res(iobs) = bad_res

		  base = base + flen
		  CYCLE ftrloop 

	      ENDIF

* Get and save box sizes for all feature coordinates. Feature-mask applied later.
*
* Weighted computation:
*   For trajectories the box size is the  great-circle-distance along the line between 
*   lon-lat points, For integrals, convert to meters; routine returns distance in Km.
*   In Z and T use distance between, construct boxes based on coordinate data.

              IF (orientation .EQ. x_dim) THEN
	         CALL BOXES_DSG (orientation, flen, unit, 
     .		   dsg_linemem(dsg_coord_lm)%ptr(base+1), 
     .		   dsg_linemem(cxdsg_coord_lm(y_dim))%ptr(base+1), box)

              ELSE
	         CALL BOXES_DSG (orientation, flen, unit, 
     .		   dsg_linemem(dsg_coord_lm)%ptr(base+1), 
     .		   dummy, box)

              ENDIF

              IF (unit .NE. 1.) THEN
	        DO i = 1, flen
		    box(i) = unit*box(i)
                 ENDDO
              ENDIF

* Large box sizes along trajectories - big gaps within the path -
* are replaced by the surrounding box size

              IF (orientation .EQ. x_dim) 
     .		 CALL MEAN_STD (box, flen, bad_val4, .TRUE., boxmean, boxstd)

* ... loop over the observations within each feature

	      iobs = base
	      oloop: DO i = 1, flen
	         iobs = iobs + 1   ! index in the contig ragged array

*    box size. For trajectory datasets, look for extra-large boxes
*    and instead use nearby delta 

		 boxdel = box(i)
		 
		 IF ( orientation.EQ.x_dim ) THEN
		    IF (boxstd .GT. 0.) THEN
		       IF ( boxdel .GT. 2.* boxstd) THEN
		          IF (i.GT.1) THEN
		             boxdel = MIN( boxstd, box(i-1) )
		          ELSEIF (i.LT.flen) THEN
		             boxdel = MIN( boxstd, box(i+1) )
		          ELSE
		             boxdel = boxmean
		          ENDIF
		       ENDIF
	            ELSE
		       res(iobs) = 0.
		       CYCLE oloop
		    ENDIF
	         ENDIF

		 res(iobs) = boxdel
	      ENDDO oloop


	      base = base + flen

	   ENDDO ftrloop

	ELSE  

* *** a simple list of instances:
* compress along the instance (feature) axis resulting in just a single value
* Each feature weighted equally.

	   res(1) = 0.

	   DO ifeature = 1, nfeatures 

* ... have the user's constraints removed this feature?
	      IF (.NOT.fmask(ifeature)) CYCLE

	      res(ifeature) = 1.
           ENDDO

	ENDIF  ! obs vs feature 

	
	RETURN
	END
