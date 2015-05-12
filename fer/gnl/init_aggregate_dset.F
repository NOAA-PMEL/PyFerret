	SUBROUTINE INIT_AGGREGATE_DSET (dname, dtitle, agg_dset, 
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
*
* Programmer Ansley Manke
* NOAA/PMEL, Seattle, WA - TMAP 
*
* V680 5/12 *acm* 6D Ferret, changes for DEFINE DATA/AGGREGATE
* V685 4/13 *acm* Ability to define ensemble with /HIDDEN
*                 Get grid of user-vars, allow more combinations with user vars
*                 better error handling.
* V695 2/15 *acm* Aggregation in either E or F direction, as defined by agg_dim
* V695+ 5/15 *sh* Call GET_AXIS_SPEEDS (pulled out of CHECK_MEMBER_DSETS)

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
	CHARACTER*(*)	dname, dtitle
	INTEGER		agg_dset, agg_dim, status
	LOGICAL		agg_quiet, agg_hide

* local variable declarations:
	LOGICAL	match, has_uvars, need_tf_lag_t
	INTEGER	TM_LENSTR, TM_LENSTR1, STR_SAME, NCF_DELETE_DSET,
     .		NCF_ADD_AGG_MEMBER, NCF_GET_AGG_MEMBER_INFO, 
     .          KNOWN_GRID,
     .		iset, dset, item, ii, nT,
     .		perm(nferdims), nsets, member_sets(maxdsets), 
     .		imatch, nv, grid, igrid, idim, ivar, nvars, iline, 
     .		slen, memb_var, memb_dset, memb_grid, ugrid, 
     .		type, fvar, ndims, ngatts, recdim, varid, attoutflag,
     .          tax_match, tf_lag_t, memb_tax, prev_line_start
	CHARACTER vname*128, buff*64, tcoordname*16
	CHARACTER TM_STRING*13
	REAL*8  TM_WORLD, avg_lag

* initialize
	need_tf_lag_t = .TRUE.
	tf_lag_t = unspecified_int4  ! lag time axis if FMRC

* check to see if this name matches an already-open dataset name.
* if so, thats an error

	DO 100 iset = 1, maxdsets
           match = ds_name(iset) .EQ. dname
           IF ( match ) GOTO 5100
  100	CONTINUE

* See if requested datasets are open and if not, try to open them.

	CALL CHECK_MEMBER_SETS (nsets, member_sets, status)
	IF (status .NE. ferr_ok .OR. nsets .EQ. 0) GOTO 9950

* Ok, all the member datasets are open and available

* Assign aggregate data set number, exiting if maximum number 
* of data sets are open

	DO 200 iset = 1,maxdsets
	  IF (ds_name(iset) .EQ. char_init1024) THEN
	    agg_dset = iset
	    GOTO 220
	  ENDIF
  200	CONTINUE

* Too many data sets are open
	CALL TM_ERRMSG ( merr_dsetlim, status, 'INIT_AGGREGATE_DSET',
     .			 no_descfile, no_stepfile,
     .			 'MAX='//TM_STRING(DBLE(maxdsets)),
     .			 no_errstring, *9950)
 220	CONTINUE

* Create datset. Will delete it later if there are no aggregate variables found.
	CALL CREATE_AGG_DSET (agg_dset, dname, dtitle,
     .			 nsets, agg_dim, iline, status)  
	IF (status .NE. ferr_ok) GOTO 5300

* perform initializations needed on a new dataset
!* interpret special parameters in the descriptor
!	CALL GET_DSET_PARMS( agg_dset, status )
!	IF ( status .NE. ferr_ok ) GOTO 5300XX

* determine relative access speeds along each axis
	CALL GET_AXIS_SPEED( agg_dset )

!* build completed ( with time axis ) grids for computable variables
!	CALL MAKE_CVAR_GRIDS( dset, status )
!	IF ( status .NE. ferr_ok ) GOTO 5300XX

* 1st member dset sets pattern for all to follow
	dset = member_sets(1)

* If this is a Forecast aggregation create name for the 2D time array
* Make it unique as necessary by adding "_" chars
	IF (agg_dim.EQ.f_dim) THEN
	   tcoordname = 'TF_TIMES'
	   slen = 8
 250	   CONTINUE
	   DO ivar = 1, maxvars
	      IF (ds_var_setnum(ivar) .EQ. dset) THEN
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

*********  top of variable loop ***************
* See what variables are shared by all the datsets
	nvars = 0
	DO 800 ivar = 1, maxvars
           IF ( ds_var_setnum(ivar) .NE. dset ) GOTO 800
	   vname = ds_var_code(ivar)
	   imatch = 1
	   DO 400 ii = 2, nsets
	      iset = member_sets(ii)
              DO 300 nv = 1, maxvars
	         IF ( (ds_var_setnum(nv) .EQ. iset) .AND.
     .                (STR_SAME(ds_var_code(nv), vname) .EQ. 0) ) THEN
                    imatch = imatch + 1
		 ENDIF
  300	      CONTINUE

* See if a user-defined LET/D variable name matches
              DO 350 nv = 1, maxvars
	         IF ( (uvar_dset(nv) .EQ. iset) .AND.
     .                (STR_SAME(uvar_name_code(nv), vname) .EQ. 0) ) THEN
                    imatch = imatch + 1
	      ENDIF
  350	      CONTINUE

  400	   CONTINUE

           IF (imatch .NE. nsets) THEN
	      IF ( .NOT.agg_quiet ) 
     .	         CALL WARN('Exclude variable from aggregate. '//
     .	         'Does not appear in all member datasets: '//
     .	         vname)
	      GOTO 800 ! Variable not in all datasets
	   ENDIF

           grid = ds_grid_number(ivar)
	   IF (grid_line(agg_dim, grid) .NE. 0) GOTO 5400

* Check that grids match for file variables. 
* NOTE if the variables in the member datasets share the same grid, and
*      there is a record axis, then they may have different timesteps but 
*      grid_line and ds_grid_start, ds_grid_end will have the same contents.
*      How to find that case? (check line_mem) Or just allow it?

* For a FMRC aggregation, the time axes must be the same units and origin 
* and length but will in general NOT share time steps.  The first time step 
* should increase from one set to the other. Not checking for this at the 
* moment.

           imatch = 1
	   prev_line_start = 0
	   IF (grid_line(t_dim, grid) .NE. 0) prev_line_start = 
     .          line_start(grid_line(t_dim, grid))
	   has_uvars = .FALSE.

           DO 600 ii = 2, nsets
	      iset = member_sets(ii)
	      tax_match = 0
              DO 500 nv = 1, maxvars
	         IF ( (ds_var_setnum(nv) .EQ. iset) .AND. 
     .                (STR_SAME(ds_var_code(nv), vname) .EQ. 0) ) THEN
                    igrid = ds_grid_number(nv)
		    DO 550 idim = 1, nferdims
		       IF (agg_dim.EQ.f_dim .AND. idim.EQ.t_dim) GOTO 550
		       IF ( grid_line(idim, grid) .NE.  
     .                      grid_line(idim, igrid) ) GOTO 500
  550	            CONTINUE
                    imatch = imatch + 1
		 ENDIF

	         IF ( (uvar_dset(nv) .EQ. iset)   .AND. 
     .                (STR_SAME(uvar_name_code(nv), vname) .EQ. 0) ) THEN

	            CALL GET_UVAR_GRID( nv, iset, status )
	            IF ( status .NE. ferr_ok ) GOTO 5600
	            igrid = KNOWN_GRID(iset,cat_user_var,nv)

	            IF (igrid .NE. grid) THEN
		    DO 555 idim = 1, nferdims
		       IF (agg_dim.EQ.f_dim .AND. idim.EQ.t_dim) GOTO 555
		       IF ( grid_line(idim, grid) .NE.  
     .                      grid_line(idim, igrid) ) GOTO 500
  555	            CONTINUE
                    ENDIF

		    imatch = imatch + 1 

		    has_uvars = .TRUE.
	         ENDIF
		 
* For forecast-direction aggregations, the time axis of the aggregation 
* is not taken directly from the ensemble members.  Instead it
* is merely a lag axis -- number of time steps and lag per step
		 IF (agg_dim .EQ. f_dim .AND. need_tf_lag_t) THEN
	            memb_tax = grid_line(t_dim, grid)
	            nT = line_dim(memb_tax)
	            avg_lag = ( TM_WORLD(nT, grid, t_dim, box_middle )
     .	                      - TM_WORLD(1,  grid, t_dim, box_middle ) )
     .			      /(nT-1)
	            CALL TM_NEW_LINE_NAME ( 'TF_LAG_T', buff )
	            CALL TM_MAKE_BASIC_AXIS( buff, 0.0D0, avg_lag, nT,
     .					     tf_lag_t, status )
	            line_unit_code(tf_lag_t) = line_unit_code(memb_tax)
	            line_units(tf_lag_t) =
     .					un_name(line_unit_code(tf_lag_t))
!	            line_tunit(tf_lag_t) =
!     .					un_convert(line_unit_code(tf_lag_t))
	            line_direction(tf_lag_t) = 'TT'
	            line_dattype  (tf_lag_t) = nf_double
	            line_dim_only (tf_lag_t) = .FALSE.  ! ?? why?
	            need_tf_lag_t = .FALSE.
		 ENDIF
		       
  500	      CONTINUE
  600	   CONTINUE
  

*  variable in all datsets but they dont have the same grid.
	   IF (imatch .NE. nsets) THEN
	      IF ( .NOT.agg_quiet ) 
     .	         CALL WARN('Exclude variable from aggregate. '//
     .	         'Grid differs in member datasets: '//
     .	         vname)
              IF (agg_dim.EQ.f_dim .AND. tax_match.NE.0) THEN
	      IF ( .NOT.agg_quiet ) 
     .	         CALL WARN('FMRC Time axes do not share units or size;'//
     .	         ' or starting times not increasing '//
     .	         vname)
              ENDIF
	      GOTO 800 
	   ENDIF
	      
* same grid for this variable in all datasets 
	   
           nvars = nvars + 1

	   CALL DEFINE_AGGREGATE_GRID (agg_dset, tcoordname, nvars, vname,
     .           ivar, iline, tf_lag_t, grid, agg_dim, member_sets, nsets)
	   IF (status .NE. ferr_ok) GOTO 5000

  800   CONTINUE

* See if user-defined (LET/D=) variables from the first dataset are 
* shared by all the datsets

	dset = member_sets(1)
	DO 1800 ivar = 1, maxvars
           IF ( uvar_dset(ivar) .NE. dset ) GOTO 1800
	   vname = uvar_name_code(ivar)
	   imatch = 1

	   grid = unspecified_int4
	   DO 1400 ii = 2, nsets
	      iset = member_sets(ii)
              DO 1300 nv = 1, maxvars
	         IF ( (ds_var_setnum(nv) .EQ. iset) .AND.
     .                (STR_SAME(ds_var_code(nv), vname) .EQ. 0) ) THEN
                    imatch = imatch + 1
		    grid = ds_grid_number(nv)
		    fvar = nv

		 ENDIF
 1300	      CONTINUE

* See if a user-defined LET/D variable name matches
              DO 1350 nv = 1, maxvars
	         IF ( (uvar_dset(nv) .EQ. iset) .AND.
     .                (STR_SAME(uvar_name_code(nv), vname) .EQ. 0) ) THEN

                    imatch = imatch + 1
		 ENDIF
 1350	      CONTINUE

 1400	   CONTINUE

	   IF (imatch .NE. nsets) THEN
	      
	      IF ( .NOT.agg_quiet ) 
     .	         CALL WARN('Exclude variable from aggregate. '//
     .	         'Does not appear in all member datasets: '//
     .	         vname)
               GOTO 1800 ! Variable not in all datasets
	   ENDIF

* If one dataset contains the variable then we will be able to define the grid.
	   
	   IF (grid .NE. unspecified_int4) THEN

              nvars = nvars + 1

	      CALL DEFINE_AGGREGATE_GRID (agg_dset, tcoordname, nvars, vname,
     .           fvar, iline, tf_lag_t, grid, agg_dim, member_sets, nsets)
	      IF (status .NE. ferr_ok) GOTO 5000

* Do not include variables that are entirely user-defined.
	   ELSE
	      IF ( .NOT.agg_quiet ) 
     .	         CALL WARN('Exclude variable from aggregate. '//
     .	         'Must be a file variable in at least '//
     .		 'one member dataset: '// vname)

	   ENDIF

 1800	CONTINUE

* If no variables from the datasets match, delete this dataset. 

	if (nvars .EQ. 0) goto 5500

* Add aggregate members to aggregate dataset

	DO 2000 iset = 1, nsets
	   status = NCF_ADD_AGG_MEMBER (agg_dset, member_sets(iset)) 
 2000	CONTINUE

	IF (dtitle(1:2) .NE. char_init) THEN
	   ds_title(agg_dset) = dtitle
	ELSE
	   ds_title(agg_dset) = 'Ensemble'
	ENDIF
	ds_mod_title(agg_dset) = ' '
	ds_time_axis(agg_dset) = ds_time_axis(member_sets(1))
	ds_format(agg_dset) = ds_format(member_sets(1))
 
* set use counts and clean up temporary grids and axes for agg. dset
	CALL TM_GARB_COL_GRIDS(agg_dset)

* If user requested /HIDE, then mark the member datasets for
* hidden status on SHOW DATA commands
	
	IF (agg_hide) THEN
 	   DO 2100 iset = 1, nsets
	      ds_hide (iset) = .TRUE. 
 2100	   CONTINUE
	ENDIF

 5000   RETURN	

* Errors: too-many datasets open so aggregate dataset can't be opened,
* or error on a member dataset. 
* ?? If some member datasets were opened, close them??)
* Return no-dataset-number.

 9950	agg_dset = unspecified_int4
	status = ferr_TMAP_error
	GOTO 5000
 9960	CALL CANCEL_DATA_SET( agg_dset, .FALSE., status )
	agg_dset = unspecified_int4
	GOTO 5000

 5100   CALL ERRMSG( ferr_invalid_command,status,
     .		'dataset name already in use: '
     .		//dname(:TM_LENSTR1(dname)), *5000 )
     
 5300	CALL ERRMSG( ferr_aggregate_error, status, ' ', *9950 )

 5400	CALL ERRMSG( ferr_TMAP_error, status,
     .		' Variables in member datasets already '
     .		//'have axis in the E direction', *9950 )

 5500	CALL ERRMSG( ferr_aggregate_error, status,
     .		'No valid datasets or datasets share no variables. ', 
     .		*9960 )

 5600	CALL ERRMSG( ferr_aggregate_error, status,
     .		'Could not get grid for user-defined variables '
     .		//uvar_name_code(nv)(:TM_LENSTR1(uvar_name_code(nv))), 
     .		*9960 )
	END

