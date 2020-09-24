	SUBROUTINE MASKED_DSG_RANGE (dset, cx, idim, lo, hi)


* If there are constraints on a DSG dataset, that affects the region in all
* directions, e.g. list/x= /y= affects the time range of data as well.
* Return the data range after applying the current context with any constraints
* This routine defines mask-array sizes and calls a subroutine to do the work.
* V76  4/20 *acm*  use the dsg_xlate_grid virtual grid with DSG coordinate variables

        IMPLICIT NONE

        include 'tmap_dims.parm'
        include 'ferret.parm'
        include 'errmsg.parm'
        include 'xcontext.cmn'
        include 'xdset_info.cmn_text'
        include 'xdsg_context.cmn'

	INTEGER dset, cx, idim
	REAL	lo, hi

	INTEGER TM_DSG_NFEATURES, TM_DSG_NF2FEATURES, grid, nfeatures, 
     .		nobs, nftrsets
	REAL	min_obs, max_obs, low, hig

	grid = dsg_xlate_grid(dset)
	nfeatures = TM_DSG_NFEATURES(grid)

	nftrsets = 0
	IF (dsg_orientation(dset) .EQ. f_dim) 
     .     nftrsets  = TM_DSG_NF2FEATURES(dset)

* max_obs, length of longest feature 
	CALL DSG_COORD_LIMS(dset, pdsg_row_size, min_obs, max_obs)
	nobs = max_obs

	CALL MASKED_DSG_RANGE_SUB (dset, cx, grid, nfeatures, nftrsets,
     .         nobs, idim, lo, hi)

	RETURN
	END

