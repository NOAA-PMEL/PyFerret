	SUBROUTINE MASKED_DSG_RANGE_SUB ( dset, cx, grid, nfeatures, 
     .					  nftrsets, nobs, idim, lo, hi)

* Return the lo and hi range for this coordinate direction after constraints
* applied.  
* V76  4/20 *acm*  Return longitude range using smaller of longitude or lon360

        IMPLICIT NONE

        include 'tmap_dims.parm'
        include 'ferret.parm'
        include 'xcontext.cmn'
        include 'xtm_grid.cmn_text'
        external xgt_grid_data
        include 'xdyn_linemem.cmn_text'
        include 'xdset_info.cmn_text'
        include 'xdsg_context.cmn'

	INTEGER dset, cx, grid, nfeatures, nftrsets, nobs, idim
	REAL	lo, hi

	LOGICAL process_feature(nfeatures), process_obs(nobs), instance_coord
	INTEGER	line, ifeature, iobs, base, fobs, sum, nobsf, 
     .		ivar, row_size_lm, coord_lm, count, instance_ndim
	REAL    GET_LINE_COORD, low, hig, low360, hig360, coord, bad

* features included in the current masking. If all features included and
* no constraints, leave lo, hi unchanged.

	CALL MAKE_DSG_FEATURE_MASK(dset, cx, process_feature, nfeatures)

	sum = 0
	DO ifeature = 1, nfeatures
	   IF (process_feature(ifeature)) sum = sum + 1
	ENDDO

	IF (cx.NE.cx_none .AND. 
     .    (cxdsg_no_coord_constraints .AND. sum.EQ.nfeatures) ) GOTO 5000

	line = grid_line(idim, dsg_xlate_grid(dset))
	IF (line .EQ. mnormal) GOTO 5000

	instance_coord = line_dim(line) .EQ. nfeatures
	instance_ndim = nfeatures

* For TimeseriesProfile datasets, the X-Y coordinates are 
* in the timeseries-index direction and the T coordinate
* is in the features direction.

	IF (nftrsets .GT. 0) THEN
	   IF (line_dim(line) .EQ. nftrsets) THEN
	      instance_coord = .TRUE.
	      instance_ndim = nftrsets
	   ELSEIF (line_dim(line) .EQ. nfeatures) THEN
	      instance_coord = .TRUE.
	      instance_ndim = nfeatures
	   ENDIF
	ENDIF

	row_size_lm = dsg_loaded_lm(dsg_row_size_var(dset))
	IF (idim .LT. e_dim) THEN
	   ivar = dsg_coord_var(idim,dset)
	   coord_lm = dsg_loaded_lm(ivar) 
	   bad = cxdsg_bad_val(idim)
	ELSE
	   coord_lm = 0
	   bad = 0
	ENDIF

	low = arbitrary_large_val8
	hig = arbitrary_small_val8
	low360 = arbitrary_large_val8
	hig360 = arbitrary_small_val8

	base = 0     ! obs index at end of preceding feature
	fobs = 0     ! count of masked data in each feature
	count = 0    ! any data in the mask?

	DO ifeature = 1, instance_ndim

	   nobsf = dsg_linemem(row_size_lm)%ptr(ifeature)
	   IF (instance_coord) nobsf = 1

	   IF (process_feature(ifeature)) THEN

	      IF (instance_coord) THEN
	         fobs = fobs + 1
		 count = count + 1
		 coord = ifeature
		 IF (idim .LT. e_dim) coord = dsg_linemem(coord_lm)%ptr(fobs)

		 IF (coord .EQ. bad) CYCLE
		 low = MIN(low, coord)
	         hig = MAX(hig, coord)  ! what about ftr_hi
		 IF (idim .EQ. x_dim) THEN
		    IF (coord .LT. 0.) coord = coord+360.
		    low360 = MIN(low360, coord)
	            hig360 = MAX(hig360, coord)
		 ENDIF

	      ELSE

* ... define observation-level mask for this feature

	         CALL MAKE_DSG_OBS_MASK (dset, cx, ifeature, base, 
     .					 process_obs, nobsf)

	         DO iobs = 1, nobsf
	            fobs = fobs+1
		    IF (process_obs(iobs)) THEN

		       count = count + 1
		       coord = dsg_linemem(coord_lm)%ptr(fobs)
		       
		       IF (coord .EQ. bad) CYCLE
		       low = MIN(low, coord)
		       hig = MAX(hig, coord)
		       IF (idim .EQ. x_dim) THEN
		          IF (coord .LT. 0.) coord = coord+360.
		          low360 = MIN(low360, coord)
	                  hig360 = MAX(hig360, coord)
		       ENDIF

	            ENDIF
		 ENDDO  ! iobs

	      ENDIF  ! instance_coord or not
	   
	   ELSE ! process_feature

	      fobs = fobs + nobsf

	   ENDIF ! process_feature
	   base = base + nobsf
	ENDDO

	IF (count .GT. 0) THEN
	   lo = low
	   hi = hig

	   IF (idim .EQ. x_dim) THEN
	      IF (hig360-low360 .LT. hi-lo) THEN
	         lo = low360
		 hi = hig360
	      ENDIF
	   ENDIF

	ENDIF

 5000	RETURN
	END
