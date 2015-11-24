	SUBROUTINE DEFINE_EF_AGGREGATE_GRID (agg_dset,
     .		tcoordname, nvars, vname, fvar, iline,
     .		tf_lag_t, grid, agg_dim, member_sets, nsets)

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
	INTEGER agg_dset, nvars, fvar, iline, tf_lag_t, grid,
     .		agg_dim, member_sets(*), nsets
	CHARACTER*(*) vname, tcoordname

* local variable declarations:
	LOGICAL	its_dynamic, itsa_uvar, has_edges
	INTEGER STR_SAME, TM_LENSTR1,
     .		big_cx, igrid, idim, gnum, nv, type, ii, iset, slen, 
     .		ndim, vtype, attype, attoutflag, i2, status,
     .		npts, nT, nF, ipt, iline2, gnum2, it2d,
     .		axcalt, axlagf, gg, ipt1, ipte, fax
	REAL*8  TDEST_WORLD

* Getting info back from the attribute-handling structure
	CHARACTER dimname*128, buff*128, name1*128
	INTEGER varid, tline, g2d

* initialize
      tline = unspecified_int4

* Define aggregate grid

      igrid = unspecified_int4
      IF (grid .NE. unspecified_int4) THEN
         igrid = 0
         CALL TM_NEXT_TMP_GRID (igrid, 710)
  710    CALL TM_ALLO_TMP_GRID (igrid, status)
         IF (status .NE. ferr_ok) GOTO 5000

         DO 700 idim = 1, nferdims
            grid_line(idim, igrid) = grid_line(idim,grid)
  700    CONTINUE
	 tline = grid_line(t_dim, igrid)
         grid_line(agg_dim, igrid) = iline
* for FMRC agg the shared T axis is a time lag 
	 IF (tf_lag_t .NE. unspecified_int4)
     .			grid_line(t_dim, igrid) = tf_lag_t 

* Make up a unique grid name for aggregate grid
         IF (agg_dim .EQ. t_dim) grid_name(igrid) = 'GEN'
         IF (agg_dim .EQ. f_dim) grid_name(igrid) = 'GFN'
         CALL TM_NEW_GRID_NAME( grid_name(igrid), buff )
         grid_name(igrid) = buff

	 gnum = igrid
      ENDIF

* ds_var slot, fill with aggregate variable in aggregate dataset.
      itsa_uvar = .FALSE.
      DO 720 nv = 1, maxvars
	      IF ( STR_SAME(ds_var_code(nv),char_init) .EQ. 0 ) GOTO 725
 720	CONTINUE
 725	ds_var_setnum(nv)   = agg_dset
	CALL string_array_modify(ds_var_code_head, nv, 
     .         ds_var_code(fvar), LEN(ds_var_code(fvar)))
	ds_var_title(nv)    = ds_var_title(fvar)
	IF ( ds_var_title(nv) .EQ. char_init128) 
     .       ds_var_title(nv)  = ds_var_code(nv)
	ds_var_titl_mod(nv) = ds_var_titl_mod(fvar)
	ds_var_type(nv)     = ds_var_type(fvar)
	ds_var_units(nv)    = ds_var_units(fvar)
	ds_missing_flag(nv) = bad_val4
	ds_bad_flag(nv)     = bad_val4
	cd_scale_factor(nv) = 1.
	cd_add_offset(nv)   = 0.
	cd_scaled(nv)       = .FALSE.

	IF (grid .NE. unspecified_int4) THEN
	   ndim = 1
	   DO 730 idim = 1, nferdims
	      ds_grid_start(idim,nv)= ds_grid_start(idim,fvar)
	      ds_grid_end(idim,nv)  = ds_grid_end(idim,fvar)
	      ds_ordering(idim,nv)  = ds_ordering(idim,fvar)
	      IF (ds_ordering(ndim,fvar) .GT. 0) ndim = ndim + 1
 730	   CONTINUE
	   ds_grid_start(agg_dim,nv)= 1
	   ds_grid_end(agg_dim,nv)  = nsets
	   ds_ordering(ndim,nv)     = agg_dim
           
	   ds_grid_number(nv) = gnum
	ENDIF

* Type should be NCFLOAT or NCCHAR
	IF (ds_var_type(nv) .EQ. ptype_string) THEN
	   type = NCCHAR
	ELSE
	   type = NCFLOAT
	ENDIF
      
	CALL CD_ADD_VAR( agg_dset, vname, nvars, 
     .         ds_var_title(nv), ds_var_units(nv), type, 
     .         .FALSE., ds_bad_flag(nv), status)
        IF (nvars .EQ. 1) name1 = vname

* If this is a forecast aggregation, then the 2D time var has been defined (or 
* will be, just below). Add a coordinates attribute to variables in the FMRC 
* dataset which now have a 2D forecast-time coordinate variable. 
* Also add an attribute (name to be finalized), _ftime2dRef pointing to TIME2D
	      
	IF (agg_dim.EQ.f_dim .AND. grid_line(t_dim, gnum).NE.int4_init) THEN
	   buff = " "
	   slen = 0
	   DO 2200 idim = nferdims, 1, -1 
	      IF (grid_line(idim, gnum) .NE. int4_init) THEN
	         dimname = line_name(grid_line(idim, gnum))
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
	   CALL CD_PUT_NEW_ATTR (agg_dset, varid, '_ftime2dRef', attype, 
     .             slen, attoutflag, tcoordname, 0., status )

	ENDIF

* Add aggregate-grid info to the lists in aggregate datset and
* aggregate-grid list.   Mark user-variables with a grid number of -999

	DO 750 ii = 1, nsets
	   iset = member_sets(ii)
	   DO 740 nv = 1, maxvars
	      IF ( (ds_var_setnum(nv) .EQ. iset) .AND. 
     .             (STR_SAME(ds_var_code(nv), vname) .EQ. 0) ) THEN
	         igrid = ds_grid_number(nv)
		 vtype = cat_file_var ! file var
	         CALL CD_ADD_AGG_VAR_INFO (agg_dset, vname, ii, vtype, 
     .             iset, igrid, iline, nv, status)
	      ENDIF
	      IF ( (uvar_dset(nv) .EQ. iset) .AND. 
     .             (STR_SAME(uvar_name_code(nv), vname) .EQ. 0) ) THEN
	         igrid = unspecified_int4
		 vtype = cat_user_var! user var
	         CALL CD_ADD_AGG_VAR_INFO (agg_dset, vname, ii, vtype, 
     .             iset, igrid, iline, nv, status)
	      ENDIF
 740	   CONTINUE
 750	CONTINUE

 
************* FMRC setup **************************************************
* Define 2D time variable and then the special axes needed for FMRC dataset
* The code that follows should be executed only once for an FMRC aggregation
* We use the test of whether the tcoordname variable is already defined
* to bail out if the code has already been run 
* (cleaner code might be possible by removing the code below to a separate
* subroutine that is called just once before processing the individual vars
* of the aggregations)
	IF (agg_dim.EQ.f_dim .AND. tline.NE.unspecified_int4) THEN

	   CALL CD_GET_VAR_ID (agg_dset, tcoordname, varid, status)
	   IF (status .EQ. ferr_ok) GOTO 5000  ! BAIL OUT -- already defined

	   igrid = 0
	   CALL TM_NEXT_TMP_GRID (igrid, 760)
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
	         grid_line(idim, g2d) = 0
	         ds_grid_start(idim,nv) = 1
	         ds_grid_end(idim,nv) = 1
	         ds_ordering(idim,nv) = 0
 780	      CONTINUE

	      grid_line(t_dim, g2d) = grid_line(t_dim,gnum)
	      grid_line(f_dim, g2d) = iline

	      ds_grid_start(t_dim,nv) = 1
	      ds_grid_end(t_dim,nv) = line_dim(tline)
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

	      iset = 0
	      vtype = cat_agg_f_var
	      CALL CD_ADD_AGG_VAR_INFO (agg_dset, tcoordname, ii, vtype, 
     .             iset, igrid, iline, nv, status)
 790	   CONTINUE

* Give the time2D variable _CoordinateAxisType and calendar attributes

	   CALL CD_GET_VAR_ID (agg_dset, tcoordname, varid, status)
	   attype = NCCHAR
	   CALL CD_PUT_NEW_ATTR (agg_dset, varid, '_CoordinateAxisType', 
     .       attype, 4, attoutflag, 'Time', 0., status )
	   slen = TM_LENSTR1(line_cal_name(tline))
	   attoutflag = 1
	   attype = NCCHAR
	   CALL CD_PUT_NEW_ATTR (agg_dset, varid, 'calendar', attype, 
     .       slen, attoutflag, line_cal_name(tline), 0., status )

* define two special axes needed for FMRC regridding:
* the 1D verification time axis and the 1D forecast
* lead time axis
* Assemble the 2D time array in line_mem and call CD_MAKE_FMRC_AXES
* Save it at the start of available line storage memory
* Also save the time box edge positions in line_mem above the 2d array
* line_mem contents will be:  2d_times, 2d_lo_lims, hi_lim_of_last_time
	   nT = line_dim(grid_line(t_dim,gnum))
	   nF = line_dim(grid_line(f_dim,gnum))
	   npts = nT * nF
	   ipt1 = next_line_mem_pos
	   ipt  = ipt1
	   IF ((ipt+2*npts+1).GT.maxlinestore) GOTO 5100

* ... loop over the member data sets, copying their time steps
	   DO ii = 1, nsets
	      CALL CD_GET_AGG_VAR_INFO (agg_dset, vname, ii, 
     .                vtype, iset, gnum2, iline2, nv, status)
	      IF (status .NE. ferr_ok) GOTO 5200
* ...  iline2 should be equal to iline -- no importance
* set up to translate all times to the T0 of the first member
	      CALL TDEST_WORLD_INIT (gnum2, grid, t_dim, status)
	      IF (status .NE. ferr_ok) RETURN 
	      DO i2 = 1, nT
	         line_mem(ipt) = TDEST_WORLD(i2, gnum2, t_dim, box_middle)
	         line_mem(ipt+npts) =
     .				 TDEST_WORLD(i2, gnum2, t_dim, box_lo_lim)
	         ipt = ipt + 1
	      ENDDO
	   ENDDO
	   line_mem(ipt1+2*npts) =
     .				 TDEST_WORLD(nT, gnum2, t_dim, box_hi_lim)

* save the 2D time step array in c memory cache
* to be recovered as needed by FIND_MEM_VAR
	   CALL CACHE_FULL_ARRAY(line_mem(ipt1), npts,
     .						ds_var_c_cache_ptr(it2d))

!***** TEST ****
!	   CALL RECOVER_CACHED_ARRAY(line_mem(ipt1+72), npts,
!     .						ds_var_c_cache_ptr(varid))


* make the two special axes 
* (varid still holds the id of the tccordname 2d time array)
	   has_edges = .TRUE.
	   CALL CD_MAKE_FMRC_AXES (agg_dset, has_edges, varid, tcoordname,
     .			ds_var_type(nv), npts, nT, nF,
     .			axcalt, axlagf, status)
	   IF (status .NE. merr_ok) GOTO 5000

* define the F calendar axis -- similar drill to above
	   ipt1 = next_line_mem_pos
	   IF ((ipt1+2*nF+1).GT.maxlinestore) GOTO 5100
	   ipt  = ipt1

* ... loop over the member data sets, copying the 1st time step of each
	   DO ii = 1, nsets
	      CALL CD_GET_AGG_VAR_INFO (agg_dset, vname, ii, 
     .                vtype, iset, gnum2, iline2, nv, status)
	      IF (status .NE. ferr_ok) GOTO 5200
	      CALL TDEST_WORLD_INIT (gnum2, grid, t_dim, status)
	      IF (status .NE. ferr_ok) RETURN 
	      line_mem(ipt) = TDEST_WORLD(1, gnum2, t_dim, box_middle )
	      line_mem(ipt+nF) = TDEST_WORLD(1, gnum2, t_dim, box_lo_lim )
	      ipt = ipt + 1
	   ENDDO
	   line_mem(ipt+nF) = TDEST_WORLD(1, gnum2, t_dim, box_hi_lim )

* ... make the forecast calendar axis
	   CALL CD_MAKE_FMRC_FCAL_AXIS (agg_dset, iline, 
     .					nF, axcalt, status)
	   IF (status .NE. merr_ok) GOTO 5000

! *sh* code in SHOW_DATA_SET_VARS indicates that I do not understand how the
! current code handles grids of aggregated datasets.  This block of code
! seems to be causing troubles ... causing SHOW DATA to display limits of
! 0:-7777 for the F axis ... so I removed the block below

!* as a precaution, a hack to correct any ds_start and ds_end
!* values that may have been made using the old iline definition 
!	   DO nv = 1, maxvars
!	      IF ( ds_var_setnum(nv) .EQ. agg_dset ) THEN
!	         gg  = ds_grid_number(nv)
!	         fax = grid_line(f_dim,gg)
!	         IF (fax .EQ. iline) THEN
!	            ds_grid_start(f_dim,nv) = line_mem(ipt1)
!	            ds_grid_end  (f_dim,nv) = line_dim(ipte)
!	         ENDIF
!	      ENDIF
!	   ENDDO

	ENDIF  ! Forecast aggregations speacial needs



 5000	RETURN
 5100	CALL ERRMSG(ferr_insuff_memory,status,
     .	        'unable to define forecast aggregation', *5000)
 5200	CALL ERRMSG(ferr_internal, status, 'define_ef_aggregate_grid!',
     .		 *5000)
	END
