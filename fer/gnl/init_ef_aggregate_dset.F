	SUBROUTINE INIT_EF_AGGREGATE_DSET (memb_dsetname_var,
     .	   nsets, have_expr, dname, dpath, dtitle, agg_dset, 
     .     agg_quiet, agg_hide, agg_dim, status )
*
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

* initialize a new ensemble (E) or forecast (F) aggregated dataset
* Aggregation member variables may be a blend of file variables and LET/D vars
* Since the user may alter LET/D definitions after the aggregation has been
* defined, the following approach is taken to processing:
*  1) during DEFINE DATA/AGG the grids of all member variables are determined
*     This means that IS_UVAR_GRID is called for LET/D vars.
*     These grid numbers are stored in the aggregation LIST structures
*     The grids of the variables in the aggregated (parent) dataset are also
*     determined and stored in ds_grid_number as if they are file variables.
*     These grids are "static" as with file variables -- always valid. 
*  2) (TBD as of this writing!!!)==> the use counts of *all* member grids are
*     bumped, so that they cannot be deleted, even if LET/D definitions change
*  3) (TBD as of this writing!!!)==> at get_grid time all of the components
*     of the aggregation are checked to
*     ensure they have a grid (IS_UVAR_GRID) -- allowing for last-minute
*     LET/D var changes
*  4) during evaluation (in IS_STRIP and IS_AGGREGATE_GATHER) the variable
*     name and member dataset number are used to make a fresh determination of
*     the variable category and grid.  Error out if grid number has changed

* Programmer Ansley Manke
* NOAA/PMEL, Seattle, WA - TMAP 
*
* V680 5/12 *acm* 6D Ferret, changes for DEFINE DATA/AGGREGATE
* V685 4/13 *acm* Ability to define ensemble with /HIDDEN
*                 Get grid of user-vars, allow more combinations with user vars
*                 better error handling.
* V695 2/15 *acm* Aggregation in either E or F direction, as defined by agg_dim
* V695+ 5/15 *sh* Call GET_AXIS_SPEEDS (pulled out of CHECK_MEMBER_DSETS)
* V695+ 10/15 *sh* rename this routine to INIT_EF_AGGREGATE_DSET
* V698 12/15 *sh* reorganization of code in order to accommodate:
*		  checks of uvar grids delayed until eval
*	          allow for LET/D def changes after DEFINE/AGG (hopefully)
*		  allow dataset list passed as a Ferret string variable
*                 allow forecast datasets out of order (sort em)
*		  allow gaps in the forecast series (insert dummy datasets)
* V698 1/16  *sh* pass explicit path name (for "USE file.agg" compatibility)
*      2/16  *sh* automatically xfer attributes from member vars to agg vars
*      3/16  *sh* trap simple invalid commands
*                 start TBOXLO[L=1], not TBOX[L=1], of tf_lag_t axis at 0.0 
* V71 12/16 *acm* For ticket 2158: working with true monthly time axes.

* arguments:
*	memb_dsetname_var - if have_expr, the variable of names (input)
*	nsets		- number of datasets in the aggregation (input)
*       have_expr	- flag if filenames were listed or in an expr (input)
*	dname		- dataset name to assign to the aggregation (input)
*	dpath		- dataset path name to assign to aggregation (input)
*	dtitle		- dataset title (input)
*	dset_num	- number assigned to this data set (output)
*       agg_quiet       - flag whether to report certain conditions (input)
*	agg_hide        - flag whether to hide the component datasets (input)
*	agg_dim		- axis of the aggregation (E or F) (input)
*	status		- return status (output)

      IMPLICIT NONE
	include 'netcdf.inc'
        include 'tmap_errors.parm'
#include "tmap_dset.parm"
        include 'tmap_dims.parm'
	include	'ferret.parm'
	include 'xdset_info.cmn_text'
	external xdset_info_data
	include 'xdset_info.cd_equiv'
	include 'xunits.cmn_text'
        include 'xprog_state.cmn'
	include 'xtm_grid.cmn_text'  
	include 'errmsg.parm'
	include 'xvariables.cmn'
	include 'xcontext.cmn'
	include 'xfr_grid.cmn'

* calling argument declarations:
	CHARACTER*(*)	dname, dpath, dtitle
	INTEGER		nsets, agg_dset, agg_dim, status
	LOGICAL		have_expr, agg_quiet, agg_hide
	REAL		memb_dsetname_var(*)

* parameter definitions
	INTEGER	pmaxgaps, pmaxvarsperset
	PARAMETER (pmaxgaps = 10,
     .		   pmaxvarsperset = 400)

* local variable declarations:
	LOGICAL	OK_AS_AGG_VAR, ITSA_TRUEMONTH_AXIS, match, has_uvars
	INTEGER	TM_LENSTR, TM_LENSTR1, STR_SAME,
     .		NCF_ADD_AGG_MEMBER, KNOWN_GRID,
     .		iset, dset1, item, ii, nT, max_str_len,
     .		perm(nferdims), member_sets(nsets+pmaxgaps),
     .		imatch, nv, grid, igrid, idim, ivar, nvars, nuvars, iline, 
     .		slen, memb_var, memb_dset, memb_grid, ugrid, nsets_wgaps,
     .		type, fvar, ndims, ngatts, recdim, varid, attoutflag,
     .          tf_lag_t, memb_tax,
     .		ref_grid, ref_fvar, ref_fvars(pmaxvarsperset),
     .		num_indices, j, jj, ignored_status
	CHARACTER buff*64, tcoordname*16
	CHARACTER TM_STRING*13
	CHARACTER*512 GET_STRING_ELEMENT_1D_X, memb_dsetname(nsets)
	CHARACTER*128 vname
	REAL*8  TM_WORLD, avg_lag

* initialize
	tf_lag_t = unspecified_int4  ! lag time axis if FMRC
	max_str_len = 512
	nsets_wgaps = nsets
	DO iset = 1, nsets+pmaxgaps
	   member_sets(iset) = pdset_dummy
	ENDDO

* must have at least two datasets for an F aggregation
	IF (agg_dim.EQ.F_dim .AND. nsets.LT.2) GOTO 5050

* check to see if this new dataset name matches an already-open dataset name.
* if so, thats an error
	DO iset = 1, maxdsets
           match = ds_name(iset) .EQ. dname
           IF ( match ) GOTO 5100
	ENDDO

* set up array of dataset identifier strings
*   these may be string dataset numers or names,  pre-init'ed or not
	IF (have_expr) THEN
*  ... dataset list passed through a string variable
	  DO iset = 1, nsets
	    memb_dsetname(iset) = GET_STRING_ELEMENT_1D_X(
     .			nsets,
     .                  memb_dsetname_var,iset,
     .			max_str_len,slen )
	  ENDDO
	ELSE
*  ... dataset names given by user on command line
	  DO iset = 1, nsets
	    memb_dsetname(iset) = cmnd_buff(item_start(iset):item_end(iset))
	  ENDDO
	ENDIF

* See if requested datasets are open and if not, try to open them.
	CALL CHECK_MEMBER_SETS (agg_dim, nsets, memb_dsetname,
     .				member_sets, status)
	IF (status .NE. ferr_ok) GOTO 9950

* forecast aggregation datasets may need to be sorted and gaps detected
	IF (agg_dim .EQ. f_dim) THEN
	   CALL SORT_FMRC_DSETS (nsets, member_sets, nsets+pmaxgaps,
     .				 nsets_wgaps, nT, status)
	   IF (status .NE. ferr_ok) GOTO 5000
	ENDIF

* Assign aggregate data set number, exiting if maximum number 
* of data sets are open
	DO iset = 1,maxdsets
	  IF (ds_name(iset) .EQ. char_init1024) THEN
	    agg_dset = iset
	    GOTO 220
	  ENDIF
	ENDDO
* ... too many data sets are open
	CALL TM_ERRMSG ( merr_dsetlim, status, 'INIT_EF_AGGREGATE_DSET',
     .			 no_descfile, no_stepfile,
     .			 'MAX='//TM_STRING(DBLE(maxdsets)),
     .			 no_errstring, *9950)
 220	CONTINUE

* Create datset. Will delete it later if there are no aggregate variables found
* Note: the pattern will the first dataset given.  They may get reordered.
	CALL CREATE_AGG_DSET (agg_dset, dname, dpath, dtitle,
     .			 nsets_wgaps, member_sets(1), agg_dim, iline, status)  
	IF (status .NE. ferr_ok) GOTO 5300

* perform initializations needed on a new dataset
* determine relative access speeds along each axis
	CALL GET_AXIS_SPEED( agg_dset )

* 1st member dset sets pattern for all to follow
	dset1 = member_sets(1)

* If this is a Forecast aggregation create name for the 2D time array
* Make it unique as necessary by adding trailing "_" chars
	IF (agg_dim.EQ.f_dim) THEN
	   tcoordname = 'TF_TIMES'
	   slen = 8
 250	   CONTINUE
	   DO ivar = 1, maxvars
	      IF (ds_var_setnum(ivar) .EQ. dset1) THEN
	         IF ( STR_SAME(tcoordname,ds_var_code(ivar)) .EQ. 0 ) THEN
	            tcoordname = tcoordname(:slen)//'_'
	            slen = slen + 1
	            GOTO 250
	         ENDIF
	      ENDIF
	   ENDDO
	ELSE
	   tcoordname = ' '
	ENDIF

*****
* find variables that are acceptable for aggregation using dset1 as the basis
* an aggregatable var must have at least member dset that is an fvar - ref_fvar
* note LET/D names that duplicate file variable names take precedence, so
*     check them first
	nuvars = 0
!	DO nv = 1, max_uvar      ! brute-force equivalent loop
	CALL deleted_list_get_undel(uvar_num_items_head,
     .                              deleted_list_result(1),
     .                              max_uvar, num_indices )
	DO j = 1, num_indices
	   nv = deleted_list_result(j)
	   IF (uvar_dset(nv) .EQ. dset1) THEN
	      IF (OK_AS_AGG_VAR(nv, cat_user_var, dset1, member_sets,
     .		  nsets_wgaps, agg_dim, agg_quiet, ref_fvar, status)  ) THEN
	         nuvars = nuvars + 1
	         ref_fvars(nuvars) = ref_fvar
	      ELSE
	         IF (status .NE. ferr_ok) GOTO 5000
	      ENDIF
	   ENDIF
	ENDDO
	nvars = nuvars

* now find the file variables
	ref_grid = unspecified_int4
        DO nv = 1, maxvars
	   IF (ds_var_setnum(nv) .EQ. dset1) THEN
* ... skip it if it duplicates a name already found as a LET/D variable
	      DO ii = 1, nuvars
	         IF (STR_SAME(ds_var_code(ref_fvars(ii)),
     .			      ds_var_code(nv)           ) .EQ. 0) GOTO 300
	      ENDDO
	      IF (OK_AS_AGG_VAR(nv, cat_file_var, dset1, member_sets,
     .		  nsets_wgaps, agg_dim, agg_quiet, ref_fvar, status)  ) THEN
	         nvars = nvars + 1
	         ref_fvars(nvars) = ref_fvar
	      ELSE
	         IF (status .NE. ferr_ok) GOTO 5000
	      ENDIF
	   ENDIF
 300	   CONTINUE
	ENDDO
*****

* If no variables from the datasets match, delete this dataset. 
	IF (nvars .EQ. 0) GOTO 5500

* For forecast aggregations, the time axis of the aggregation 
* is not taken directly from the ensemble members.  Instead it
* is merely a lag axis -- number of time steps and lag per step
	IF (agg_dim .EQ. f_dim) THEN
	   ref_grid = ds_grid_number(ref_fvars(1))
	   IF (ref_grid .EQ. unspecified_int4) GOTO 5400
	   memb_tax = grid_line(t_dim, ref_grid)
	   avg_lag = ( TM_WORLD(nT, ref_grid, t_dim, box_middle )
     .	             - TM_WORLD(1,  ref_grid, t_dim, box_middle ) )
     .			/(nT-1)
	   CALL TM_NEW_LINE_NAME ( 'TF_LAG_T', buff )
! pre 3/16	   CALL TM_MAKE_BASIC_AXIS( buff, 0.0D0, avg_lag, nT,
* note that if the FMRC gets aggregated by the THREDDS Data Server this
* code never gets called.  See CD_SCAN_AUX_COORDS for that case.
* If the member axes are true-month axes, make the time axis of the 
* aggregation a daily axis, not true-month.
	   CALL TM_MAKE_BASIC_AXIS( buff, avg_lag/2., avg_lag, nT,
     .				    tf_lag_t, status )
	   line_unit_code(tf_lag_t) = line_unit_code(memb_tax)
	   IF (ITSA_TRUEMONTH_AXIS(memb_tax)) line_unit_code(tf_lag_t) = pun_day
	   line_units(tf_lag_t) = un_name(line_unit_code(tf_lag_t))
	   line_direction(tf_lag_t) = 'TT'
	   line_dattype  (tf_lag_t) = nf_double
	   line_dim_only (tf_lag_t) = .FALSE.  ! indicates netCDF dim w/out var
	ENDIF


* register the list of variables found suitable for this aggregation
	ii = 0
	nv = 0
	DO WHILE (ii .LT. nvars)
	   ii = ii + 1
	   nv = nv + 1! kludgy: incremented just below when TF_TIMES is added
	   ref_fvar = ref_fvars(ii)
	   vname = ds_var_code(ref_fvar)
	   grid = ds_grid_number(ref_fvar)
	   CALL DEFINE_EF_AGGREGATE_GRID (agg_dset, tcoordname, nv, vname,
     .					  ref_fvar, iline, tf_lag_t, grid,
     .					  agg_dim, member_sets, nsets_wgaps)
	   IF (status .NE. ferr_ok) GOTO 5000

* transfer to the aggregated variable whatever special attributes it inherits
	   CALL SET_AGG_VAR_ATTS(agg_dset, vname, nv, nsets_wgaps, 
     .			         ref_fvar, member_sets, status)
	   IF (status .NE. ferr_ok) GOTO 9960
	ENDDO

* register member datasets to aggregate dataset
	DO iset = 1, nsets_wgaps
	   status = NCF_ADD_AGG_MEMBER (agg_dset, iset, member_sets(iset)) 
	ENDDO
 
* remove duplicate temp grid definitions amd makegrid names unique
	CALL AGG_CLEAN_GRIDS(agg_dset, agg_dim, iline, tf_lag_t)

* set use counts and remove temp status of agg grids and axes
	CALL TM_GARB_COL_GRIDS(agg_dset)

* set the official time axis of the dataset
	CALL CD_GET_TIME_AXIS( agg_dset, t_dim, ignored_status )
	IF (ignored_status .NE. merr_ok) CALL WARN
     .					  ('INIT_EF_GET_TIME_AX crptn')

* If user requested /HIDE, then mark the member datasets for hidden status
	IF (agg_hide) THEN
 	   DO iset = 1, nsets_wgaps
	      ds_hide (member_sets(iset)) = .TRUE. 
	   ENDDO
	ENDIF

* successful completion
	RETURN

* error bail out
* remove all hidden member datasets of the failed aggregation
* ... this is a little brute force in that it may cancel a previously opened
* ... dataset included using DEFINE DATA/AGG/HIDE (havnt checked carefully)
 5000	DO ii = 1, nsets_wgaps
	   iset = member_sets(ii)
	   IF (iset .EQ. pdset_dummy
     .    .OR. iset .EQ. unspecified_int4) CYCLE
	   IF (ds_name(iset) .EQ. char_init1024) CYCLE ! already deleted
	   IF (ds_hide (iset)) CALL CANCEL_ONE_DATA_SET( iset, ignored_status )
	ENDDO
 6000	RETURN	

* Errors: too-many datasets open so aggregate dataset can't be opened,
* or error on a member dataset. 
* ?? If some member datasets were opened, close them??)
* Return no-dataset-number.

 9950	agg_dset = unspecified_int4
	status = ferr_TMAP_error
	GOTO 5000
* wipe out the partially-created aggregated dataset
 9960	CALL TM_PURGE_TMP_GRIDS(ignored_status)
	CALL TM_CLOSE_SET_COND_DEALLO( agg_dset, .FALSE., ignored_status )
	agg_dset = unspecified_int4
	GOTO 5000

 5050	CALL ERRMSG( ferr_invalid_command,status,
     .		'cannot F-aggregate a single member dataset', *6000 )

 5100   CALL ERRMSG( ferr_invalid_command,status,
     .		'dataset name already in use: '
     .		//dname(:TM_LENSTR1(dname)), *5000 )
     
 5300	CALL ERRMSG( ferr_aggregate_error, status, ' ', *9950 )


 5400	CALL ERRMSG( ferr_aggregate_error, status,
     .		'No time axis for aggregation',          *9960 )

 5500	CALL ERRMSG( ferr_aggregate_error, status,
     .		'No valid datasets or datasets share no variables. ', 
     .		*9960 )

	END

