        SUBROUTINE DEFINE_F_AGGREGATE_GRID(agg_dset, tcoordname, 
     .     nvars, vname, fvar, iline, agg_lag_t, ref_grid, 
     .     member_sets, nsets, status)

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
* Define ensemble or forecast axis and grid using that axis.
* For uvars, grid is set to
* unspecified_int4 when calling this. Will defer the grid check for later.

* Programmer Ansley Manke
* NOAA/PMEL, Seattle, WA - TMAP 
*
* V680 5/12 *acm* 6D Ferret, changes for DEFINE DATA/AGGREGATE
* V695+ 4/15 *acm* and *sh* -- added support for forecast aggregations
* V698 1/16 *sh* - support gappy and inconsistent length forecast members
* V698 3/16 *sh* - shift both lag axes and F-calendar to pts at low cell edge
* V710 *acm* 10/16 Ticket 2422: variables uvar_grid, uvar_data_type, uvar_aux_cat, 
*                  uvar_aux_var to be replaced with calls to LIST routines in NCF_Util.c 
* v71  12/16 *acm* Ticket 2158: working with true monthly time axes.
* v72   6/17 *acm* Variable type in aggregations should match what is in the member files
* v74  10/18 *kms* Split E and F aggregation routines

      IMPLICIT NONE
        include 'tmap_errors.parm'
#include "tmap_dset.parm"
        include 'tmap_dims.parm'
	include	'ferret.parm'
	include 'xdset_info.cmn_text'
	external xdset_info_data
	include 'xdset_info.cd_equiv'
        include 'xprog_state.cmn'
	include 'xtm_grid.cmn_text'  
	include 'errmsg.parm'
	include 'xvariables.cmn'
	include 'xcontext.cmn'
	include 'xfr_grid.cmn'
	include 'interp_stack.parm'
	include 'netcdf.inc'

* calling argument declarations:
	INTEGER agg_dset, nvars, fvar, iline, agg_lag_t, ref_grid,
     .		member_sets(*), nsets, status
	CHARACTER*(*) vname, tcoordname

* local variable declarations:
        LOGICAL NC_GET_ATTRIB, ITS_FMRC_2DTIME, TM_DFPEQ, 
     .          its_dynamic, itsa_uvar, has_edges, has_gaps,
     .          found_fvar, ef_ordered_time2d, true_month
	INTEGER STR_SAME, TM_LENSTR1,
     .		big_cx, igrid, idim, agg_grid, nv, type, ii, iset, slen, 
     .		ndim, vtype, i2, i,
     .		npts, nT, nF, lagF, ipt, iline2, memb_grid, it2d,
     .		gap_size, max_gap_size, l,
     .		axlagt, axcalt, axlagf, axcalf, gg, ipt1, ipte, fax,
     .		nv_dummy, igrid_dummy, vtype_dummy, num_indices,
     .		ef_time2d_dset, ef_time2d_grid, iaxis, fvartype,
     .		time2dline, superline, tevery, fevery
	REAL*8  TDEST_WORLD, TM_WORLD_AX, GET_LINE_COORD, 
     .		tcoord, tedge, tstart, fcoord, delta, tend

* Getting info back from the attribute-handling structure
        INTEGER maxlen
        PARAMETER (maxlen = 128)
	CHARACTER*(maxlen) dimname, buff, name1
	INTEGER varid, t2did, tline, g2d, globalid, ref_id, attlen,
     .		attype, attoutflag
        REAL    attvals(10)

* initialize
	tline = unspecified_int4
	has_gaps = .FALSE.
	ef_ordered_time2d = .FALSE.
	ef_time2d_dset = ds_var_setnum(fvar)
	time2dline = unspecified_int4

* Get the file-variable type
        slen = TM_LENSTR1(vname)
	varid = 0
        CALL CD_GET_VAR_TYPE (ef_time2d_dset, varid, vname(:slen), 
     .                        fvartype, status)

* Define aggregate grid

      igrid = unspecified_int4
      IF (ref_grid .NE. unspecified_int4) THEN
	 CALL TM_ALLO_TMP_GRID (igrid, status)
         IF (status .NE. ferr_ok) GOTO 5000

         DO 700 idim = 1, nferdims
            grid_line(idim, igrid) = grid_line(idim,ref_grid)
 700     CONTINUE
	 tline = grid_line(t_dim, igrid)
	 IF (ef_ordered_time2d) THEN
	    ef_time2d_grid = ref_grid  ! grid of fvar (w/out agg axis)
	 ELSE
	    grid_line(f_dim, igrid) = iline
	 ENDIF
* for FMRC agg the shared T axis is a time lag 
	 IF (agg_lag_t .NE. unspecified_int4) THEN
	    grid_line(t_dim, igrid) = agg_lag_t 
	 ENDIF
* Make up a unique grid name for aggregate grid
         grid_name(igrid) = 'GFN'
! as temps all have same name -- duplicates are resolved in AGG_CLEAN_GRIDS
!         CALL TM_NEW_GRID_NAME( grid_name(igrid), buff )
!         grid_name(igrid) = buff

	 agg_grid = igrid
      ENDIF

* ds_var slot, fill with aggregate variable in aggregate dataset.
      itsa_uvar = .FALSE.
      DO 720 nv = 1, maxvars
	      IF ( STR_SAME(ds_var_code(nv),char_init) .EQ. 0 ) GOTO 725
 720	CONTINUE
 725	ds_var_setnum(nv)   = agg_dset
	CALL string_array_modify(ds_var_code_head, nv, 
     .         ds_var_code(fvar), LEN(ds_var_code(fvar)))
! ?? better: inherit these attibutes from members in SET_AGG_VAR_ATTS
	ds_var_title(nv)    = ds_var_title(fvar)
	IF ( ds_var_title(nv) .EQ. char_init128) 
     .       ds_var_title(nv)  = ds_var_code(nv)
	ds_var_titl_mod(nv) = ds_var_titl_mod(fvar)
	ds_var_units(nv)    = ds_var_units(fvar)
	ds_missing_flag(nv) = ds_missing_flag(fvar)	! was bad_val4
	ds_bad_flag(nv)     = ds_bad_flag(fvar)		! was bad_val4
	ds_var_type(nv)     = ds_var_type(fvar)
	cd_scale_factor(nv) = 1.
	cd_add_offset(nv)   = 0.
	cd_scaled(nv)       = .FALSE.

	IF (ref_grid .NE. unspecified_int4) THEN
	   ndim = 1
	   DO 730 idim = 1, nferdims
	      ds_grid_start(idim,nv)= ds_grid_start(idim,fvar)
	      ds_grid_end(idim,nv)  = ds_grid_end(idim,fvar)
	      ds_ordering(idim,nv)  = ds_ordering(idim,fvar)
	      IF (ds_ordering(ndim,fvar) .GT. 0) ndim = ndim + 1
 730	   CONTINUE
           ds_grid_start(f_dim,nv) = 1
           ds_grid_end(f_dim,nv) = nsets
           ds_ordering(ndim,nv) = f_dim  ! last dimension of the grid
* ... limit T axis length to the T length of the F aggregation
           IF (agg_lag_t.NE.unspecified_int4)
     .		ds_grid_end(t_dim,nv) = line_dim(agg_lag_t)

	   ds_grid_number(nv) = agg_grid
	ENDIF

* Type should be what is in the original file
	IF (ds_var_type(nv) .EQ. ptype_string) THEN
	   type = NCCHAR
	ELSE
	   type = fvartype
	ENDIF

* Hook the aggregated variable into the linked list structures
	CALL CD_ADD_VAR( agg_dset, vname, nvars, 
     .         ds_var_title(nv), ds_var_units(nv), type, 
     .         .FALSE., ds_bad_flag(nv), status)
        IF (nvars .EQ. 1) name1 = vname

* If this is a forecast aggregation, then the 2D time var has been defined 
* (or will be, just below). Add a coordinates attribute to variables 
* in the FMRC dataset which now have a 2D forecast-time coordinate variable. 
* Also add an attribute _ftime2d pointing to TIME2D
	     
	IF (grid_line(t_dim, agg_grid).NE.int4_init) THEN
	   buff = " "
	   slen = 0
	   DO 2200 idim = nferdims, 1, -1 
	      IF (grid_line(idim, agg_grid) .NE. int4_init) THEN
	         dimname = line_name(grid_line(idim, agg_grid))
		 IF (idim .EQ. t_dim) dimname = tcoordname
	         IF (slen.EQ.0) THEN
		    buff = dimname(:TM_LENSTR1(dimname) )
		 ELSE
 	            buff = buff(:slen)//' '//dimname(:TM_LENSTR1(dimname) )
                 ENDIF
		 slen = TM_LENSTR1(buff)
	      ENDIF
 2200	   CONTINUE
	   CALL CD_GET_VAR_ID (agg_dset, vname, varid, status)
	   slen = TM_LENSTR1(buff)
	   attoutflag = 1
	   attype = NCCHAR
	   CALL CD_PUT_NEW_ATTR (agg_dset, varid, 'coordinates', attype, 
     .             slen, attoutflag, buff, 0., status )

	   slen = TM_LENSTR1(tcoordname)
	   attoutflag = 1
	   attype = NCCHAR
	   CALL CD_PUT_NEW_ATTR (agg_dset, varid, '_ftime2d', attype, 
     .             slen, attoutflag, tcoordname, 0., status )

	ENDIF

* Add aggregate-grid info to the lists in aggregate datset and aggregate-grid list.
* Mark user-variables with a grid number of -999.  Note if there is a LET/D variable
* that duplicates the name of a file var it is the file var that gets registered,
* even though the LET/D var has precedence during evaluation.  This allows for the
* possibility that the LET/D var may be cancelled.  But it also allows the possibility
* that the grid of the fvar is unacceptable -- a chance of a crash for a pathological
* dataset.
	dsets: DO ii = 1, nsets
	   iset = member_sets(ii)
* ... special hack for the TF_TIMES 2d time array when it is E-aggregated
	   IF (ef_ordered_time2d) THEN
	      vtype = cat_ef_time2d
	      CALL CD_ADD_AGG_VAR_INFO (agg_dset, vname, ii, vtype, 
     .                ef_time2d_dset, ef_time2d_grid, iline, fvar, status)
	      CYCLE dsets
	   ENDIF
	   found_fvar = .FALSE.
	   IF (iset .NE. pdset_dummy) THEN
* ... add this member data set as an fvar as part of this aggregated variable
	      DO nv = 1, maxvars
	         IF ( (ds_var_setnum(nv) .EQ. iset) .AND. 
     .                (STR_SAME(ds_var_code(nv), vname) .EQ. 0) ) THEN
	            igrid = ds_grid_number(nv)
		    vtype = cat_file_var
	            CALL CD_ADD_AGG_VAR_INFO (agg_dset, vname, ii, vtype, 
     .                iset, igrid, iline, nv, status)
	            found_fvar = .TRUE.
	            EXIT
	         ENDIF
	      ENDDO
	      IF (.NOT.found_fvar) THEN
* ... add this member data set as a LET/D as part of this aggregated variable
* ... optimize search using deleted_list_get_undel
!	         DO nv = 1, max_uvar      ! brute-force equivalent loop
		 CALL deleted_list_get_undel(uvar_num_items_head,
     .                              deleted_list_result(1),
     .                              max_uvar, num_indices )
	         DO i = 1, num_indices
	            nv = deleted_list_result(i)
	            IF ( uvar_num_items(nv) .EQ. uvar_deleted) CYCLE
	            IF ( (uvar_dset(nv) .EQ. iset) .AND. 
     .                   (STR_SAME(uvar_name_code(nv), vname) .EQ. 0) ) THEN
		       vtype = cat_user_var

		       CALL GET_SAVED_UVAR_GRID(nv, iset, igrid)
	               CALL CD_ADD_AGG_VAR_INFO (agg_dset, vname, ii, vtype, 
     .                   iset, igrid, iline, nv, status)
* ... bump the use count for this grid to protect it from being deleted if
*     the user changes the LET/D definition -- grid is needed during the
*     gather process (IS_STRIP & IS_AGGREGATE_GATHER).
*  ==> be sure to un-use (deallo) it when the aggregation gets canceled
*  ==> or if an error occurs prior to the complete initialization of this agg
	               CALL TM_USE_DYN_GRID(igrid)
	               EXIT
	            ENDIF
	         ENDDO
	      ENDIF
	   ELSE
* ... add the dummy data set as a part of this variable 
*     Note that the first member of the agg by common sense cannot be a dummy
	      CALL CD_ADD_AGG_VAR_INFO (agg_dset, vname, ii, vtype_dummy, 
     .             pdset_dummy, igrid_dummy, iline, nv_dummy, status)
	   ENDIF
* ... grab information from the first dataset to reuse with dummy datasets 
	   IF (ii .EQ. 1) THEN
*    ... OK_AS_AGG_VAR guarantees grid is always valid -- even for LET/D vars
	      IF (igrid .EQ. unspecified_int4)
     .           STOP 'DEFINE_F_AGGREGATE_GRID grid err'
	      nv_dummy = nv
	      igrid_dummy = igrid
	      vtype_dummy = cat_file_var
	   ENDIF
	ENDDO dsets

 
************* FMRC setup **************************************************
* Define 2D time variable and then the special axes needed for FMRC dataset
* The code that follows should be executed only once for an FMRC aggregation
* We use the test of whether the tcoordname variable is already defined
* to bail out if the code has already been run.
* (Cleaner code might be possible by removing the code below to a separate
* subroutine that is called just once before processing the individual vars
* of the aggregations.)

	IF (tline.NE.unspecified_int4) THEN

	   CALL CD_GET_VAR_ID (agg_dset, tcoordname, t2did, status)
	   IF (status .EQ. ferr_ok) GOTO 4000  ! BAIL OUT -- already defined

	   igrid = 0
  760	   CALL TM_ALLO_TMP_GRID (igrid, status)
	   IF (status .NE. ferr_ok) GOTO 5000
	   CALL TM_NEW_GRID_NAME ( 'FTI', buff )
           grid_name(igrid) = buff
	   g2d = igrid

* ds_var slot, define TIME2D

	   DO 770 nv = 1, maxvars
	      IF ( STR_SAME(ds_var_code(nv),char_init) .EQ. 0 ) GOTO 775
 770	   CONTINUE
 775	   ds_var_setnum(nv)   = agg_dset
	   CALL string_array_modify(ds_var_code_head, nv, 
     .         tcoordname, LEN(tcoordname))
	   ds_var_title(nv)    = 'Forecast time step matrix'
	   ds_var_titl_mod(nv) = ' '
	   ds_var_type(nv)     = ptype_float
	   ds_missing_flag(nv) = bad_val4
	   ds_bad_flag(nv)     = bad_val4
	   cd_scale_factor(nv) = 1.
	   cd_add_offset(nv)   = 0.
	   cd_scaled(nv)       = .FALSE.
* ... re-assemble units = 'days since 15-jan-1982' from units and t0
	   slen = TM_LENSTR1(line_units(tline))
	   ds_var_units(nv)    = line_units(tline)(:slen) // ' since '
     .			      // line_t0(tline)

	   IF (g2d .NE. unspecified_int4) THEN
	      DO 780 idim = 1, nferdims
	         grid_line(idim, g2d) = 0   ! presumably =mpsnorm ??
	         ds_grid_start(idim,nv) = 1
	         ds_grid_end(idim,nv) = 1
	         ds_ordering(idim,nv) = 0
 780	      CONTINUE

* grid for TIME2D actually uses the superset axis defined later
* reassign once that has been figured out
	      grid_line(t_dim, g2d) = grid_line(t_dim,agg_grid)
	      grid_line(f_dim, g2d) = iline

	      ds_grid_start(t_dim,nv) = 1
	      ds_grid_end(t_dim,nv) = line_dim(agg_lag_t)  ! limit to agg dim
	      ds_ordering(1,nv) = t_dim

	      ds_grid_start(f_dim,nv) = 1
	      ds_grid_end(f_dim,nv) = nsets
              ds_ordering(2,nv) = f_dim

	      ds_grid_number(nv) = g2d
	   ENDIF

* ... save the sequence number of this variable, so we can cache it later
	   it2d = nv

	   type = NCDOUBLE
	   nvars = nvars + 1
      
	   CALL CD_ADD_VAR( agg_dset, tcoordname, nvars, 
     .         ds_var_title(nv), ds_var_units(nv), type, 
     .         .FALSE., ds_bad_flag(nv), status)

* Add the aggregate-grid info with CALL CD_ADD_AGG_VAR_INFO
* save the grid of the member datasets so we can get back the
* right time axis for each member, but set member dataset to 0
* because the 2d Time variable is not in the member sets.

	   DO 790 ii = 1, nsets
              iset = member_sets(ii)
	      DO 795 i2 = 1, maxvars
	         IF ( (ds_var_setnum(i2) .EQ. iset) .AND. 
     .                (STR_SAME(ds_var_code(i2), name1) .EQ. 0) ) 
     .            igrid = ds_grid_number(i2)
 795	      CONTINUE

	      iset = pagg_promotion_indicator_dset
	      vtype = cat_agg_f_var 
	      CALL CD_ADD_AGG_VAR_INFO (agg_dset, tcoordname, ii, vtype, 
     .             iset, igrid, iline, nv, status)
 790	   CONTINUE

* Give the time2D variable _CoordinateAxisType and calendar atts

	   CALL CD_GET_VAR_ID (agg_dset, tcoordname, t2did, status)
	   attoutflag = 1
	   attype = NCCHAR
	   CALL CD_PUT_NEW_ATTR (agg_dset, t2did, '_CoordinateAxisType', 
     .       attype, 4, attoutflag, 'Time', 0., status )
	   slen = TM_LENSTR1(line_cal_name(tline))
	   attoutflag = 1
	   attype = NCCHAR
	   CALL CD_PUT_NEW_ATTR (agg_dset, t2did, 'calendar', attype, 
     .       slen, attoutflag, line_cal_name(tline), 0., status )

* an FMRC needs 4 axes:
* 2 are the native axes of the data grids in compressed block FMRC form
*   - native T: a lag axis of time steps between each successive forecast
*   - native F: a calendar axis of times at which each run was initiated
* 2 are target regridding axes:
*   - target T: a calendar axis of the target times being forecast
*   - target F: a lag (lead) axis of time steps prior to the target time

* first define the two target axes
* Assemble the 2D time array in line_mem and call CD_MAKE_FMRC_AXES
* Save it at the start of available line storage memory
* Also save the time box edge positions in line_mem above the 2d array
* line_mem contents will be:  2d_times, 2d_lo_lims, hi_lim_of_last_time
	   nT = line_dim(grid_line(t_dim,agg_grid))
	   nF = line_dim(grid_line(f_dim,agg_grid))

* DYN linemem: allocate coordinate memory linemem for 2d_times and lineedg is the upper
* 

* KMS - The time axis for the FMRC then needs to be on a time axis that is a superset
* of both the forecasted (verification) times and the initialization times.
*
* Assume the forecasted and initialization time axes are regular.  I think a reasonable 
* assumption.  Also assume that an initialization time after the first forecast matches 
* a forecasted time for the first forecast.  This might be a less reasonable assumption.
* (If this becomes a problem, may have have the user provide the axis to use.)
*
* To define this superaxis, need to find the a pair of times where a forecasted time equals
* an initialization time (stepsF * deltaF == stepsT * deltaT); eg, the third forecasted 
* time is the same time as the second initialization time.  Then the delta of the superset 
* time axis is the delta of one divided by number steps for the other to the common time 
* (deltaF / stepsT == deltaT / stepsF).

* get the forecasted times for the first forecast - must exist
           CALL TM_ALLO_TMP_LINE(superline, status)
           IF ( status .NE. ferr_ok) GOTO 5200
           CALL GET_LINE_DYNMEM(nT, superline, status)
           IF ( status .NE.ferr_ok) GOTO 5100
           IF ( member_sets(1) .EQ. pdset_dummy ) 
     .        GOTO 5300
           CALL CD_GET_AGG_VAR_INFO (agg_dset, vname, 1, 
     .        vtype, iset, memb_grid, iline2, nv, status)
           IF (status .NE. ferr_ok) GOTO 5200
* ... if the first member has a LET/D var, use time axis info from another var
           IF ( memb_grid .EQ. unspecified_int4 )
     .          memb_grid = igrid_dummy
* set up to translate all times to the T0 of the first member
           CALL TDEST_WORLD_INIT (memb_grid, ref_grid, t_dim, status)
           IF (status .NE. ferr_ok) GOTO 5000 
           DO i2 = 1, nT
              tcoord = TDEST_WORLD(i2, memb_grid, t_dim, box_middle)
              CALL PUT_LINE_COORD(linemem(superline)%ptr, i2, tcoord)
           ENDDO
           tstart = GET_LINE_COORD(linemem(superline)%ptr, 1)

* Loop over the initialization times for forecasts after the first, looking for one 
* that matches a forecasted time for the first forecast.  Start with closest forecasts, 
* to (hopefully) give the smallest superset axis.
           DO ii = 2, nsets
              IF (member_sets(ii) .NE. pdset_dummy) THEN
                 CALL CD_GET_AGG_VAR_INFO(agg_dset, vname, ii, 
     .              vtype, iset, memb_grid, iline2, nv, status)
                 IF (status .NE. ferr_ok) GOTO 5200
* ... if the first member has a LET/D var, use time axis info from another var
                 IF (memb_grid .EQ. unspecified_int4) 
     .               memb_grid = igrid_dummy
* set up to translate all times to the T0 of the first member
                 CALL TDEST_WORLD_INIT(memb_grid, ref_grid, t_dim, status)
                 IF (status .NE. ferr_ok) GOTO 5000 
                 fcoord = TDEST_WORLD(1, memb_grid, t_dim, box_middle)
                 DO i2 = 2, nT
                    tcoord = GET_LINE_COORD(linemem(superline)%ptr, i2)
                    IF ( TM_DFPEQ(tcoord, fcoord) ) THEN
* deltaT = (tcoord-tstart)/(i2-1), stepsF = (ii-1), delta = deltaT/stepsF
*    or looking at it the other way (fstart == tstart):
* deltaF = (fcoord-fstart)/(ii-1), stepsT = (i2-1), delta = deltaF/stepsT
                       delta = (tcoord-tstart) / ((i2-1)*(ii-1))
                       fevery = i2 - 1 
                       tevery = ii - 1
                       GOTO 800
                    ENDIF
                 ENDDO
              ENDIF
           ENDDO
* If got here, no initialization time matched a forecasted time in the first forecast
           GOTO 5400

 800       CONTINUE
* get the last forecasted time for the last forecast - must exist
           IF ( member_sets(nsets) .EQ. pdset_dummy ) 
     .        GOTO 5500
           CALL CD_GET_AGG_VAR_INFO (agg_dset, vname, nsets, 
     .        vtype, iset, memb_grid, iline2, nv, status)
           IF (status .NE. ferr_ok) GOTO 5200
* ... if the first member has a LET/D var, use time axis info from another var
           IF ( memb_grid .EQ. unspecified_int4 )
     .          memb_grid = igrid_dummy
* set up to translate all times to the T0 of the first member
           CALL TDEST_WORLD_INIT (memb_grid, ref_grid, t_dim, status)
           IF (status .NE. ferr_ok) GOTO 5000 
           tend = TDEST_WORLD(nT, memb_grid, t_dim, box_middle)

* Remove any common denominators in fevery and tevery and factor them into delta;
* e.g., fevery=8, tevery=12, delta=0.5 ==> fevery=2, tevery=3, delta=2.0
           ii = 2
           DO WHILE ( (ii .LE. tevery) .AND. (ii .LE. fevery) )
              IF ( (MOD(fevery,ii) .EQ. 0) .AND. 
     .             (MOD(tevery,ii) .EQ. 0) ) THEN
                 fevery = fevery / ii
                 tevery = tevery / ii
                 delta = delta * ii
              ELSE IF ( ii .EQ. 2 ) THEN
*                2 -> 3
                 ii = ii + 1
              ELSE
*                3 -> 5 -> 7 -> ...
                 ii = ii + 2
              ENDIF
           ENDDO

* So want FMRC transforms put onto this superset time axis defined by tstart, tend, and delta
* A new forecast series starts every fevery steps on this axis.
* A forecast gives predictions every tevery steps on this axis.
* Undefined should remain in locations not assigned in the 2D superset time axis

* make the FMRC regrid target axes: T-calendar and F-lag
* as well as the source superset source T-lag and F-calendar axes.
* Also iline which is the actual F-calendar source axis
* (t2did still holds the id of the tcoordname 2d time array)
           CALL CD_MAKE_FMRC_AXES(agg_dset, t2did, tcoordname,
     .        ds_var_type(nv), tstart, tend, delta, fevery, tevery,
     .        nT, nF, axlagt, axcalt, axlagf, axcalf, iline, status)
           IF (status .NE. merr_ok) GOTO 5000

* Since a regular axes was assumed, can fill the 2D time array from the above information.
* This 2D array, saved in linear memory, are the times of each forecasted value (inner loop) 
* for each forecast (outer loop) - on the superset axis
           npts = nT * tevery * nF * fevery
           CALL TM_ALLO_TMP_LINE(time2dline, status)
           IF (status .NE. ferr_ok) GOTO 5200
           CALL GET_LINE_DYNMEM (npts, time2dline, status) 
           IF (status .NE. ferr_ok) GOTO 5100
           ipt = 1
           DO ii = 1, nF * fevery
              fcoord = tstart + (ii-1) * delta
              DO i2 = 1, nT * tevery
                 tcoord = fcoord + (i2-1) * delta
                 CALL PUT_LINE_COORD(linemem(time2dline)%ptr, ipt, tcoord)
                 tedge  = tcoord - 0.5 * delta
                 CALL PUT_LINE_COORD(lineedg(time2dline)%ptr, ipt, tedge)
                 ipt = ipt + 1
              ENDDO
           ENDDO
* add the final upper edge value
           tedge = tstart + (nF * fevery - 1) * delta 
     .                    + (nT * tevery - 1) * delta + 0.5 * delta
           CALL PUT_LINE_COORD(lineedg(time2dline)%ptr, npts+1, tedge)

* reset the grid for the 2D time array to use the superset axis
           grid_line(t_dim, g2d) = axlagt
           grid_line(f_dim, g2d) = axcalf
           ds_grid_end(t_dim,it2d) = nT * tevery
           ds_grid_end(f_dim,it2d) = nF * fevery

* save the 2D time step array in c memory cache
* to be recovered as needed by FIND_MEM_VAR
           CALL CACHE_FULL_ARRAY(linemem(time2dline)%ptr, npts, 
     .                           ds_var_c_cache_ptr(it2d))

* ... assign global attributes pointing to the target regrid axes
*    (CF requires variable attributes, but they are helpful as globals, too)
           CALL CD_GET_VAR_ID (agg_dset, '.', globalid, status)
           attoutflag = 1
           attype = NCCHAR
           slen = TM_LENSTR1(line_name(axcalt))
           CALL CD_PUT_NEW_ATTR (agg_dset, globalid, '_cal_t_ax', attype, 
     .             slen, attoutflag, line_name(axcalt), 0., status )
           slen = TM_LENSTR1(line_name(axlagf))
           CALL CD_PUT_NEW_ATTR (agg_dset, globalid, '_lag_f_ax', attype, 
     .             slen, attoutflag, line_name(axlagf), 0., status )
           slen = TM_LENSTR1(line_name(axcalf))
           CALL CD_PUT_NEW_ATTR (agg_dset, globalid, '_cal_f_ax', attype, 
     .             slen, attoutflag, line_name(axcalf), 0., status )
           slen = TM_LENSTR1(line_name(axlagt))
           CALL CD_PUT_NEW_ATTR (agg_dset, globalid, '_lag_t_ax', attype, 
     .             slen, attoutflag, line_name(axlagt), 0., status )

* make the regrid target axes non-temporary and set use counts to 1
           CALL TM_RE_ALLO_TMP_LINE(axcalt)
           CALL TM_USE_LINE(axcalt)
           CALL TM_RE_ALLO_TMP_LINE(axlagf)
           CALL TM_USE_LINE(axlagf)
           CALL TM_RE_ALLO_TMP_LINE(axcalf)
           CALL TM_USE_LINE(axcalf)
           CALL TM_RE_ALLO_TMP_LINE(axlagt)
           CALL TM_USE_LINE(axlagt)

	ENDIF  ! end of forecast aggregation special coordinate creation
* F-agg atts _cal_t_ax and _lag_f_a_x point to the special regridding axes
 4000   CONTINUE
          slen = TM_LENSTR1(line_name(axcalt))
          attoutflag = 1
          attype = NCCHAR
          CALL CD_PUT_NEW_ATTR (agg_dset, varid, '_cal_t_ax', attype, 
     .             slen, attoutflag, line_name(axcalt), 0., status )
          CALL CD_PUT_NEW_ATTR (agg_dset, varid, '_lag_f_ax', attype, 
     .             slen, attoutflag, line_name(axlagf), 0., status )
          CALL CD_PUT_NEW_ATTR (agg_dset, varid, '_cal_f_ax', attype, 
     .             slen, attoutflag, line_name(axcalf), 0., status )
          CALL CD_PUT_NEW_ATTR (agg_dset, varid, '_lag_t_ax', attype, 
     .             slen, attoutflag, line_name(axlagt), 0., status )

* Done with the 2D array which has been stored (or not if an error)
 5000   IF (time2dline .NE. unspecified_int4) THEN
           IF (line_allocated(time2dline) .GT. 0) 
     .        CALL FREE_LINE_DYNMEM(time2dline)
        ENDIF
        RETURN

 5100	CALL ERRMSG(ferr_aggregate_error,status,
     .	        'unable to define forecast aggregation', *5000)
 5200	CALL ERRMSG(ferr_internal, status, 'DEFINE_F_AGGREGATE_GRID!',
     .		 *5000)
 5300	CALL ERRMSG(ferr_aggregate_error, status,
     .		'no suitable file vars in 1st aggregation member', *5000)
 5400   CALL ERRMSG(ferr_aggregate_error, status,
     .          'no common forecasted and initialization time', *5000)
 5500	CALL ERRMSG(ferr_aggregate_error, status,
     .		'no suitable file vars in last aggregation member', *5000)
	END
