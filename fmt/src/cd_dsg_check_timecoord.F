	SUBROUTINE CD_DSG_CHECK_TIMECOORD (nfeatures, obs_len, row_size_lm,
     .					   t_coord_lm, errmsg, status)


* check the time data: is it monotonically increasing?  For now this is
* an error; later perhaps we'll sort unsorted data internally? This checks 
* for strict monotonic.  Could perhaps use tm_check_coords-type logic

* Also check that the row-size variable sums to the obs-dimlen

	include 'tmap_errors.parm'
	include 'tmap_dims.parm'
        include 'xdyn_linemem.cmn_text'


	INTEGER nfeatures, obs_len, row_size_lm, t_coord_lm, status
	CHARACTER*(*) errmsg

	INTEGER i, irow, base, n_in_row, iobs, nrow, tsiz
	REAL	clast, coord

* Is time in the feature-direction, e.g. for a profile dataset?
* If so no need to check time.
	tsiz = lm_size(t_coord_lm)
	IF (tsiz .EQ. nfeatures) THEN
	   status = merr_ok
	   GOTO 5000
	ENDIF

	base = 0
	DO irow = 1, nfeatures 

	   n_in_row = dsg_linemem(row_size_lm)%ptr(irow)
	   clast = 0.

	   IF ( base+n_in_row .GT. obs_len) GOTO 5200
	   DO i = 1, n_in_row
	      iobs = base + i
	      coord = dsg_linemem(t_coord_lm)%ptr(iobs)
	      IF (i.GT.1) THEN
	         IF (coord-clast .LT. 0.) GOTO 5100  ! not monotonic
	      ENDIF
	      clast = coord

	   ENDDO
	   base = base + n_in_row
	ENDDO

	IF (base .NE. obs_len) GOTO 5200

	status = merr_ok

 5000	RETURN

 5100   errmsg = 'Time coordinates are not increasing within each feature. '//
     .     'Data must be sorted by time.'
	 status = merr_dsg_grid
	GOTO 5000
 5200   errmsg = 'Row-size data must sum to the length of the observation axis.'
	status = merr_dsg_grid
	GOTO 5000

	END