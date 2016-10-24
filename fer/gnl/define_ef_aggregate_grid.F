	SUBROUTINE DEFINE_EF_AGGREGATE_GRID (agg_dset,
     .		tcoordname, nvars, vname, fvar, iline,
     .		tf_lag_t, ref_grid, agg_dim, member_sets, nsets)

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
	INTEGER agg_dset, nvars, fvar, iline, tf_lag_t, ref_grid,
     .		agg_dim, member_sets(*), nsets
	CHARACTER*(*) vname, tcoordname

* local variable declarations:
	LOGICAL	NC_GET_ATTRIB, ITS_FMRC_2DTIME, its_dynamic, itsa_uvar,
     .		has_edges, has_gaps, found_fvar, ef_ordered_time2d
	INTEGER STR_SAME, TM_LENSTR1,
     .		big_cx, igrid, idim, agg_grid, nv, type, ii, iset, slen, 
     .		ndim, vtype, i2, i, status,
     .		npts, nT, nF, lagF, ipt, iline2, memb_grid, it2d,
     .		gap_size, max_gap_size, l,
     .		axcalt, axlagf, gg, ipt1, ipte, fax,
     .		nv_dummy, igrid_dummy, vtype_dummy, num_indices,
     .		ef_time2d_dset, ef_time2d_grid
	REAL*8  TDEST_WORLD, TM_WORLD_AX, fdelta

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

* the 2D time variable in an E-aggregation of F-aggregations has special needs
* its grid should not have the agg axis 
	IF (agg_dim .EQ. E_dim) THEN
	   ef_ordered_time2d = ITS_FMRC_2DTIME(ef_time2d_dset, fvar)
	ENDIF

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
	    grid_line(agg_dim, igrid) = iline
	 ENDIF
* for FMRC agg the shared T axis is a time lag 
	 IF (tf_lag_t .NE. unspecified_int4) THEN
	    grid_line(t_dim, igrid) = tf_lag_t 
	 ENDIF
* Make up a unique grid name for aggregate grid
         IF (agg_dim .EQ. e_dim) grid_name(igrid) = 'GEN'
         IF (agg_dim .EQ. f_dim) grid_name(igrid) = 'GFN'
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
	   ds_grid_start(agg_dim,nv)= 1
	   ds_grid_end(agg_dim,nv)  = nsets
	   ds_ordering(ndim,nv)     = agg_dim  ! last dimension of the grid
* ... limit T axis length to the T length of the F aggregation
           IF (agg_dim.EQ.f_dim .AND. tf_lag_t.NE.unspecified_int4)
     .		ds_grid_end(t_dim,nv) = line_dim(tf_lag_t)

	   ds_grid_number(nv) = agg_grid
	ENDIF

* Type should be NCFLOAT or NCCHAR
	IF (ds_var_type(nv) .EQ. ptype_string) THEN
	   type = NCCHAR
	ELSE
	   type = NCFLOAT
	ENDIF

* Hook the aggregated variable into the linked list structures
	CALL CD_ADD_VAR( agg_dset, vname, nvars, 
     .         ds_var_title(nv), ds_var_units(nv), type, 
     .         .FALSE., ds_bad_flag(nv), status)
        IF (nvars .EQ. 1) name1 = vname

* If this is a forecast aggregation, then the 2D time var has been defined (or 
* will be, just below). Add a coordinates attribute to variables in the FMRC 
* dataset which now have a 2D forecast-time coordinate variable. 
* Also add an attribute (name to be finalized), _ftime2d pointing to TIME2D
	     
	IF (agg_dim.EQ.f_dim .AND. grid_line(t_dim, agg_grid).NE.int4_init) THEN
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

* Add aggregate-grid info to the lists in aggregate datset and
* aggregate-grid list.   Mark user-variables with a grid number of -999
* Note if there is a LET/D variable that duplicates the name of a file var
* it is the file var that gets registered, even though the LET/D var has
* precedence during evaluation.  This allows for the possibility thatthe LET/D
* var may be cancelled. BUT it also allows the possibility that the grid
* of the fvar is unacceptable -- a chance of a crash for a pathological dataset
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
	      IF (igrid .EQ. unspecified_int4) STOP 'DEFINE_EF_AGG grid err'
	      nv_dummy = nv
	      igrid_dummy = igrid
	      vtype_dummy = cat_file_var
	   ENDIF
	ENDDO dsets

 
************* FMRC setup **************************************************
* Define 2D time variable and then the special axes needed for FMRC dataset
* The code that follows should be executed only once for an FMRC aggregation
* We use the test of whether the tcoordname variable is already defined
* to bail out if the code has already been run 
* (cleaner code might be possible by removing the code below to a separate
* subroutine that is called just once before processing the individual vars
* of the aggregations)
	IF (agg_dim.EQ.f_dim .AND. tline.NE.unspecified_int4) THEN

	   CALL CD_GET_VAR_ID (agg_dset, tcoordname, t2did, status)
	   IF (status .EQ. ferr_ok) GOTO 4000  ! BAIL OUT -- already defined

	   igrid = 0
!	   CALL TM_NEXT_TMP_GRID (igrid, 760)   ! not needed
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

	      grid_line(t_dim, g2d) = grid_line(t_dim,agg_grid)
	      grid_line(f_dim, g2d) = iline

	      ds_grid_start(t_dim,nv) = 1
	      ds_grid_end(t_dim,nv) = line_dim(tf_lag_t)  ! limit to agg dim
	      ds_ordering(1,nv) = t_dim

	      ds_grid_start(f_dim,nv) = 1
	      ds_grid_end(f_dim,nv) = nsets
	      ds_ordering(2,nv) = agg_dim

	      ds_grid_number(nv) = g2d
	   ENDIF

* ... save the sequence number of this variable, so we can cache it later
	   it2d = nv

	   type = NCFLOAT
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

* Give the time2D variable _CoordinateAxisType, _ftime2d and calendar atts

	   CALL CD_GET_VAR_ID (agg_dset, tcoordname, t2did, status)
	   attype = NCCHAR
	   CALL CD_PUT_NEW_ATTR (agg_dset, t2did, '_CoordinateAxisType', 
     .       attype, 4, attoutflag, 'Time', 0., status )
!	   CALL CD_PUT_NEW_ATTR (agg_dset, t2did, '_ftime2d', 
!     .       attype, 4, attoutflag, 'self', 0., status )
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
	   npts = nT * nF
	   ipt1 = next_line_mem_pos
	   ipt  = ipt1
	   IF ((ipt+2*npts+1).GT.maxlinestore) GOTO 5100

* ... loop over the member data sets, copying their time steps
	   DO ii = 1, nsets
	      IF (member_sets(ii) .EQ. pdset_dummy) THEN
* ... insert a series of missing time values in the 2d time array
	         has_gaps = .TRUE.
	         DO i2 = 1, nT
	            line_mem(ipt)      = unspecified_val8
	            line_mem(ipt+npts) = unspecified_val8
	            ipt = ipt + 1
	         ENDDO
	      ELSE
* ... insert the series of time step values from this forecast
	         CALL CD_GET_AGG_VAR_INFO (agg_dset, vname, ii, 
     .                vtype, iset, memb_grid, iline2, nv, status)
	         IF (status .NE. ferr_ok) GOTO 5200
* ... iline2 should be equal to iline -- no importance; not used
* ... if the first member has a LET/D var, use time axis info from another var
	         IF (memb_grid .EQ. unspecified_int4) memb_grid = igrid_dummy
* set up to translate all times to the T0 of the first member
	         CALL TDEST_WORLD_INIT (memb_grid, ref_grid, t_dim, status)
	         IF (status .NE. ferr_ok) GOTO 5000 
	         DO i2 = 1, nT
	            line_mem(ipt) = TDEST_WORLD(i2, memb_grid, t_dim, box_middle)
	            line_mem(ipt+npts) =
     .				 TDEST_WORLD(i2, memb_grid, t_dim, box_lo_lim)
	            ipt = ipt + 1
	         ENDDO
	      ENDIF
	   ENDDO
	   line_mem(ipt1+2*npts) =
     .				 TDEST_WORLD(nT, memb_grid, t_dim, box_hi_lim)

* ... if there were gaps in the forecast time array, fill 'em
*     always compute lagF - the number of time steps between forecasts
	   CALL FILL_FMRC_TIME_GAPS(has_gaps,
     .				    line_mem(ipt1),
     .				    line_mem(ipt1+npts),
     .				    nT,nF, lagF, status)
	   IF (status .NE. ferr_ok) GOTO 5000

* save the 2D time step array in c memory cache
* to be recovered as needed by FIND_MEM_VAR
	   CALL CACHE_FULL_ARRAY(line_mem(ipt1), npts,
     .						ds_var_c_cache_ptr(it2d))

!***** TEST ****
!	   CALL RECOVER_CACHED_ARRAY(line_mem(ipt1+72), npts,
!     .						ds_var_c_cache_ptr(it2d))


* make the FMRC regrid target axes: T-calendar and F-lag 
* (t2did still holds the id of the tccordname 2d time array)
	   has_edges = .TRUE.
	   CALL CD_MAKE_FMRC_AXES (agg_dset, has_edges, t2did, tcoordname,
     .			ds_var_type(nv), npts, nT, nF,
     .			axcalt, axlagf, status)
	   IF (status .NE. merr_ok) GOTO 5000

*****
* define the F calendar axis
	   ipt1 = next_line_mem_pos
	   IF ((ipt1+2*nF+1).GT.maxlinestore) GOTO 5100

! per Andrew W., this axis has been shifted 1/2 cell to the left
! so that the F coordinate lies on the low limit (initialization time) of
! the 1st time coordinate from each forecast
! Only the coordinates need be got from the FMRC members. The bounds are
! simply midpoints between them.

* Get the F-step values -- lower bounds of the L=1 cells
*   ... copying them from the axcalt axis
	   l = 1
	   DO ipt = ipt1, ipt1+nsets-1
	      line_mem(ipt) = TM_WORLD_AX(l, axcalt, box_lo_lim )
	      l = l + lagF
	   ENDDO

* Fill in the F-cell bounds just after the F-step values
* ... we will base the axis extreme edges on the average cell width --
*     only an approximate anwser for unequally spaced monthly time units
	   Fdelta = ( line_mem(ipt1+nsets-1) - line_mem(ipt1)) / (nsets-1) 
*   ... the lower limit of the first point
	   line_mem(ipt1+nsets) = line_mem(ipt1) - Fdelta/2
*   ... the midpoints between the F-steps
	   DO ipt = ipt1+1, ipt1+nsets-1
	      line_mem(ipt+nsets) = (line_mem(ipt) + line_mem(ipt-1))/2
	   ENDDO
*   ... the upper limit of the last point
	   line_mem(ipt1+2*nsets) = line_mem(ipt1+nsets-1) + Fdelta/2

! THIS GIANT BLOCK OF COMMENTED CODE used to populate the F axis coords
! It would get the timestep values of the L=1
! output of each model run, and the corresponding cell bounds
! It has been replaced with the much simpler block of code above

!* ... similar drill to above but need to do again, because cell bounds
!*     are needed in axis definition
!* ... loop over the member data sets, inserting the 1st time step of each
!*     Both the time point and one of the cell bounds needs to be inserted
!	   max_gap_size = 0
!	   DO ii = 1, nsets
!	      IF (member_sets(ii) .NE. pdset_dummy) THEN
!	         CALL CD_GET_AGG_VAR_INFO (agg_dset, vname, ii, 
!     .                vtype, iset, memb_grid, iline2, nv, status)
!	         IF (status .NE. ferr_ok) GOTO 5200
!	         CALL TDEST_WORLD_INIT (memb_grid, ref_grid, t_dim, status)
!	         IF (status .NE. ferr_ok) GOTO 5000 
!	         line_mem(ipt)    = TDEST_WORLD(1, memb_grid, t_dim,
!     .								box_middle )
!	         line_mem(ipt+nF) = TDEST_WORLD(1, memb_grid, t_dim,
!     .								box_lo_lim )
!	         gap_size = 0
!	      ELSE
!	         gap_size = gap_size + 1
!	         IF (gap_size .GT. max_gap_size) max_gap_size = gap_size
!	         CALL RECOVER_CACHED_2D_ARRAY_POINT(line_mem(ipt), 
!     .			nT, nF, 1, ii, ds_var_c_cache_ptr(it2d))
!	         IF (gap_size .EQ. 1) THEN
!*      ... the previous forecast is still active in the loop
!	            line_mem(ipt+nF) = TDEST_WORLD(1, memb_grid, t_dim,
!     .								 box_hi_lim )
!	         ELSE
!*      ... leave a gap in the time cell bounds -- fill in right after this
!	            line_mem(ipt+nF) = unspecified_val8
!	         ENDIF
!	      ENDIF
!	      ipt = ipt + 1
!	   ENDDO
!	   line_mem(ipt+nF) = TDEST_WORLD(1, memb_grid, t_dim, box_hi_lim )
!
!* ... if there were gaps wider than 1 missing forecast, we need to fill in
!*     some missing time cell bounds
!	   IF (max_gap_size .GT. 1) THEN
!	      ii = 1
!	      ipt  = ipt1 + nF
!	      DO WHILE (ii .LT. nsets)
!	         ii  = ii  + 1
!	         ipt = ipt + 1
!	         IF (line_mem(ipt) .EQ. unspecified_val8) THEN
!*       ... fill in a gap by linear interpolation
!*           note that this is exactly right for equally spaced coordinates
!*           but only approximate for monthly.  After improved support for 
!*           monthly
!	            gap_size = 0
!	            i2 = ii+1
!	            DO WHILE (i2 .LE. nsets+1)
!	               gap_size = gap_size + 1
!	               IF (line_mem(ipt+gap_size) .NE. unspecified_val8) EXIT
!	            ENDDO
!*       ... fill the gap
!	            DO i = ipt, ipt+gap_size-1
!	               line_mem(ipt) = line_mem(ipt-1)
!     .				+ (i-ipt+1)
!     .				* (line_mem(ipt+gap_size)-line_mem(ipt-1))
!     .				/ (gap_size+1)
!	            ENDDO
!	         ENDIF
!	      ENDDO
!	   ENDIF

* ... make the native F-calendar axis from the pointstime steps  in line_mem
	   CALL CD_MAKE_FMRC_FCAL_AXIS (agg_dset, iline, 
     .					nF, axcalt, status)
	   IF (status .NE. merr_ok) GOTO 5000

* ... put attribute _CoordinateAxistype = 'RunTime' onto the F-calendar axis
! cannot put attribute onto axis, because axis isn't in dataset List structure
! this is part of a larger problem that the newly created axes of aggregations
! have not yet been inserted into any List structures ... so their attributes
! cannot be manipulated
!          CALL CD_ADD_COORD_VAR( pdset_coord_vars, stuff (add new axis) ...
!          IF ( status .NE. ferr_ok ) GOTO 5600
!	   slen = TM_LENSTR1(line_name(iline))
!	   CALL CD_GET_VAR_ID (pdset_coord_vars,
!     .		'('//line_name(iline)(:slen)//')', varid, status)
!	   attoutflag = 1
!	   attype = NCCHAR
!	   CALL CD_PUT_NEW_ATTR (pdset_coord_vars, varid, '_CoordinateAxistype',
!     .		   attype, 7, attoutflag, 'RunTime', 0., status )

* ... assign global attributes pointing to the target regrid axes
*    (CF requires variable attributes, but they are helpful as globals, too)
	   CALL CD_GET_VAR_ID (agg_dset, '.', globalid, status)
	   slen = TM_LENSTR1(line_name(axcalt))
	   CALL CD_PUT_NEW_ATTR (agg_dset, globalid, '_cal_t_ax', attype, 
     .             slen, attoutflag, line_name(axcalt), 0., status )
	   CALL CD_PUT_NEW_ATTR (agg_dset, globalid, '_lag_f_ax', attype, 
     .             slen, attoutflag, line_name(axlagf), 0., status )

* make the regrid target axes non-temporary and set use counts to 1
	   CALL TM_RE_ALLO_TMP_LINE(axcalt)
	   CALL TM_USE_LINE(axcalt)
	   CALL TM_RE_ALLO_TMP_LINE(axlagf)
	   CALL TM_USE_LINE(axlagf)

	ENDIF  ! end of forecast aggregation special coordinate creation


* F-agg atts _cal_t_ax and _lag_f_a_x point to the special regridding axes
 4000	IF (agg_dim.EQ.f_dim) THEN
	   slen = TM_LENSTR1(line_name(axcalt))
	   attoutflag = 1
	   attype = NCCHAR
	   CALL CD_PUT_NEW_ATTR (agg_dset, varid, '_cal_t_ax', attype, 
     .             slen, attoutflag, line_name(axcalt), 0., status )
	   CALL CD_PUT_NEW_ATTR (agg_dset, varid, '_lag_f_ax', attype, 
     .             slen, attoutflag, line_name(axlagf), 0., status )
	ENDIF


 5000	RETURN
 5100	CALL ERRMSG(ferr_insuff_memory,status,
     .	        'unable to define forecast aggregation', *5000)
 5200	CALL ERRMSG(ferr_internal, status, 'define_ef_aggregate_grid!',
     .		 *5000)
 5300	CALL ERRMSG(ferr_aggregate_error, status,
     .		'no suitable file vars in 1st aggregation member', *5000)
	END
