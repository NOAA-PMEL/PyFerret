	SUBROUTINE INIT_U_AGGREGATE_DSET (memb_dsetname_var,
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
* NOAA/PMEL, Seattle, WA - SDIG
*
* V7   5/12 *acm* Union aggregations: DEFINE DATA/AGG/U
* V7.2 5/17 *acm* Ticket 2526 For SHOW DATA/MEMBERS of Union datasets need to 
*                 have called NCF_ADD_AGG_MEMBERto list the dataset members in 
*                 the linked-list structure for the union dataset

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
*	agg_dim		- axis of the aggregation (is irrelevant)
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
	INTEGER	pmaxvarsperset
	PARAMETER (pmaxvarsperset = 400)

* local variable declarations:
	LOGICAL match, itsa_uvar
	INTEGER	TM_LENSTR1, STR_SAME, NCF_ADD_AGG_MEMBER,
     .		iset, dset1, item, ii, max_str_len,member_sets(nsets), 
     .		nv, nvars, iline, slen, jlen, un_vars(pmaxvarsperset),
     .		num_indices, j, member, ignored_status
	REAL	sset
	CHARACTER TM_STRING*13
	CHARACTER*512 GET_STRING_ELEMENT_1D_X, memb_dsetname(nsets)
	CHARACTER*10 TM_FMT, strset
	CHARACTER*128 vname

* initialize
	max_str_len = 512
	nsets = nsets
	DO iset = 1, nsets
	   member_sets(iset) = pdset_dummy
	ENDDO

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

* Assign aggregate data set number, exiting if maximum number 
* of data sets are open
	DO iset = 1,maxdsets
	  IF (ds_name(iset) .EQ. char_init2048) THEN
	    agg_dset = iset
	    GOTO 220
	  ENDIF
	ENDDO
* ... too many data sets are open
	CALL TM_ERRMSG ( merr_dsetlim, status, 'INIT_U_AGGREGATE_DSET',
     .			 no_descfile, no_stepfile,
     .			 'MAX='//TM_STRING(DBLE(maxdsets)),
     .			 no_errstring, *9950)
 220	CONTINUE

* Create datset. Will delete it later if there are no aggregate variables found
* Note: the pattern will the first dataset given.  They may get reordered.
	iline = unspecified_int4
	agg_dim = unspecified_int4
	CALL CREATE_AGG_DSET (agg_dset, dname, dpath, dtitle,
     .			 nsets, member_sets(1), agg_dim, iline, status)  
	IF (status .NE. ferr_ok) GOTO 5300

* 1st member dset sets pattern for all to follow
	dset1 = member_sets(1)


*****
* find variables from the first dataset and register them in the aggregation.

	iset = dset1

	nvars = 0
!	DO nv = 1, max_uvar      ! brute-force equivalent loop
	CALL deleted_list_get_undel(uvar_num_items_head,
     .                              deleted_list_result(1),
     .                              max_uvar, num_indices )
	DO j = 1, num_indices
	   nv = deleted_list_result(j)
	   IF (uvar_dset(nv) .EQ. dset1) THEN
	      nvars = nvars + 1
	      un_vars(nvars) = nv
	      vname = uvar_name_code(nv)
	      itsa_uvar = .TRUE.
	      CALL DEFINE_U_AGGREGATE_VAR (agg_dset, iset, nvars, 
     .           vname, nv, itsa_uvar, nsets, status)
              IF (status .NE. ferr_ok) GOTO 5300
	   ENDIF
	ENDDO

* now find the file variables
        DO nv = 1, maxvars
	   IF (ds_var_setnum(nv) .EQ. dset1) THEN
* ... skip it if it duplicates a name already found as a LET/D variable
	      DO ii = 1, nvars
	         IF (STR_SAME(ds_var_code(un_vars(ii)),
     .			      ds_var_code(nv)           ) .EQ. 0) GOTO 300
	      ENDDO
	      nvars = nvars + 1
	      un_vars(nvars) = nv
	      vname = ds_var_code(nv)
	      itsa_uvar = .FALSE.
	      CALL DEFINE_U_AGGREGATE_VAR (agg_dset, iset, nvars, vname, nv, 
     .         itsa_uvar, nsets, status)
           IF (status .NE. ferr_ok) GOTO 5300
	   ENDIF
 300	   CONTINUE
	ENDDO
*****

* If no variables from the dataset, delete this dataset. 
	IF (nvars .EQ. 0) GOTO 5500

* *** loop over the member datasets ***
	DO iset = 2, nsets
	   member = member_sets(iset)
	   IF (member .EQ. pdset_dummy) CYCLE ! ignore dummy (missing) dataset

*****
* find variables from the other member datasets. Add new ones to the aggregation.

	   CALL deleted_list_get_undel(uvar_num_items_head,
     .                              deleted_list_result(1),
     .                              max_uvar, num_indices )
	   DO j = 1, num_indices
	      nv = deleted_list_result(j)
	      IF (uvar_dset(nv) .EQ. member) THEN
	         nvars = nvars + 1
	         un_vars(nvars) = nv
	         vname = uvar_name_code(nv)
	         itsa_uvar = .TRUE.
	         CALL DEFINE_U_AGGREGATE_VAR (agg_dset, member, nvars, 
     .               vname, nv, itsa_uvar, nsets, status)
	         IF (status .NE. ferr_ok) GOTO 5300
	      ENDIF
	   ENDDO

* now file variables
	   DO nv = 1, maxvars
	      IF (ds_var_setnum(nv) .EQ. member) THEN
* ... skip it if it duplicates a name already found as a LET/D variable
*     Issue a note unless they said /QUIET

	         DO ii = 1, nvars
	            IF (STR_SAME(ds_var_code(un_vars(ii)),
     .			      ds_var_code(nv)           ) .EQ. 0) THEN
                       sset = member
		       slen = TM_LENSTR1(ds_var_code(nv))
		       strset = TM_FMT(sset, 10, 10, jlen)
		       
                       IF (.NOT.agg_quiet) CALL WARN ('Skipping variable '//
     .			ds_var_code(nv)(:slen)//
     .			' from set '//strset(:jlen) //
     .			', already in the Union dataset.')
		       GOTO 400
	            ENDIF
	         ENDDO
	         nvars = nvars + 1
	         un_vars(nvars) = nv
	         vname = ds_var_code(nv)
	         itsa_uvar = .FALSE.
	         CALL DEFINE_U_AGGREGATE_VAR (agg_dset, member, nvars, 
     .            vname, nv, itsa_uvar, nsets, status)
	         IF (status .NE. ferr_ok) GOTO 5300
	      ENDIF
 400	   CONTINUE
  	   ENDDO


	ENDDO  ! 2, nsets

	IF (nvars .EQ. 0) GOTO 5500


* register member datasets to aggregate dataset
	DO iset = 1, nsets
	   status = NCF_ADD_AGG_MEMBER (agg_dset, iset, member_sets(iset)) 
	ENDDO

* If user requested /HIDE, then mark the member datasets for hidden status
	IF (agg_hide) THEN
 	   DO iset = 1, nsets
	      ds_hide (member_sets(iset)) = .TRUE. 
	   ENDDO
	ENDIF

* successful completion
	RETURN

* error bail out
* remove all hidden member datasets of the failed aggregation
* ... this is a little brute force in that it may cancel a previously opened
* ... dataset included using DEFINE DATA/AGG/HIDE (havnt checked carefully)
 5000	DO ii = 1, nsets
	   iset = member_sets(ii)
	   IF (iset .EQ. pdset_dummy
     .    .OR. iset .EQ. unspecified_int4) CYCLE
	   IF (ds_name(iset) .EQ. char_init2048) CYCLE ! already deleted
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

 5100   CALL ERRMSG( ferr_invalid_command,status,
     .		'dataset name already in use: '
     .		//dname(:TM_LENSTR1(dname)), *5000 )
     
 5300	CALL ERRMSG( ferr_aggregate_error, status, ' ', *9950 )

 5500	CALL ERRMSG( ferr_aggregate_error, status,
     .		'No valid datasets ', 
     .		*9960 )

	END

