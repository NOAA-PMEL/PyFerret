	SUBROUTINE CD_CLEAN_GRIDS ( dset, cdfid, temp_axnams,
     .				    ok_share_taxis, native_tax, status )


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
*
* check for errors and eliminate duplication from newly defined grids and axes
* When the routine exits there will be no duplicate axis names although there
* will be holes in line_memory from deleted axes.
* When the routine exits there will still be duplicate grid names among the
* newly defined grids.  These may be eliminated after variables are associated.
* Finally, determine the "record" axis of this data set,

* Programmer Steve Hankin
* NOAA/PMEL, Seattle, WA - Tropical Modeling and Analysis Program

* revision 0.0 - 2/92
* v500 - 3/99 *kob* - up length of variable vname to correspond w/ increased
*                     size of line_name
*      - 4/99 *kob* - need to increase size of temp_axnams as well
*      - 7/99 *sh* - bug fix: need conosotent use of char_init16 on line_name
* V510 *sh* 3/00 - convert to use of dynamic grids and axes
*		 - use TM_GET_LINEMATCH to ensure that renamed axes also match
* V552 *acm* 5/03   increase vname and buff to 128 chars 
* V600 *ACM* attribute control. 
*             Replace NCDINQ with CD_GET_DS_DIMS
* V601 *acm* 11/06 If there is a new axis name, save the original axis name of 
*                  the axis as an attribute of the coordinate variable in the 
*                  linked-list attribute structure, and change the axis name itself
*                  in the attribute structure.
* V602 *acm* 12/06 When replacing or resetting an axis, test for the existence of
*                  the attribute orig_file_axname. If it already exists, call the
*                  routine to replace the attribute rather than trying to add a new
*                  attribute.
* V65  *acm* 1/10  Change all netcdf2-style calls and parameter names to 
*                  netcdf3-style calls, which are also consistent with netcdf4.
* V674 2/12 *acm* 6D Ferret: use nferdims rather than 4 for dimension indexing
* v694 1/15 *acm* for ticket 2227: if a dimension from a nc file is not also a 
*                 1-D coordinate var, don't write the axis Ferret creates
* V698 11/15 *sh* new args control whether sharing of T axis is allowed
* V741  6/18 *acm* small cleanup: catch error on TM_ALLO_TMP_GRID
* V751 10/19 *acm* Issue 1955: don't try to do the rename for the fix to ticket 2227
*                  if the dimension name wasn't stored as a variable to begin with.
* v7.6 2/2020 *acm* handling  files with dimensions that are not coordinate variables

* argument definitions:
*       dset    - pointer to TMAP data set
*	cdfid	- netCDF id number for already opened CDF file
*       temp_axnams - temporary storage of axis names to construct grids
*       ok_share_taxis - (input) is it ok to reuse T axis of another dataset
*       native_tax - (input) the native time axis of this dataset
*	status	- return status

* include files
	include 'tmap_errors.parm'
	include 'tmap_dims.parm'
#include "tmap_dset.parm"
#include "gt_lib.parm"
        include 'xbuild_grids.cmn'
	include 'xtm_grid.cmn_text'
	include 'xdset_info.cmn_text'
        include 'netcdf.inc'


* argument declarations
	LOGICAL ok_share_taxis
	INTEGER	 dset, cdfid, native_tax, status
*4/99 *kob*
        CHARACTER*128 temp_axnams(nferdims,max_temp_grid)

* local parameter definitions:
      INTEGER     max_grd_per_cdf
      PARAMETER ( max_grd_per_cdf = 40 )

* local variable declarations:
      LOGICAL TM_SAME_GRID_DEF, 
     .        dup_name, dup_guts, coordvar
      INTEGER TM_GET_LINEMATCH,  TM_GET_LINENUM, TM_GET_GRIDNUM, 
     .        TM_LENSTR1, STR_SAME,
     .        igrid, iaxis, count, idim, i_dup, ivar,
     .        npts, istat,  reused(max_lines:line_ceiling),
     .        varid, attid, attype, attlen, attoutflag, len,
     .        vartype, nvdims, nvatts, outflag, dimid
* upped vname to 64 - also removed the *24 from the CHARACTER
* upped vname to 128 
      CHARACTER vname*128, buff*128, attname*128
      REAL dummy, vdims(8)
	

***** AXES ******
* check for duplicate name / guts of axes
* if completely identical (name and guts) then eliminate the new one
* if new definition with already used name then flag this
	iaxis = 0	! to get start of linked list
 100	CALL TM_NEXT_TMP_LINE( iaxis, *110)
	  reused(iaxis) = TM_GET_LINEMATCH( iaxis, dup_name, dup_guts )
* ... hack: multi-file T-aggregated dset must not share any existing taxis at
*     this stage, because more time steps will be appended to it
	  IF (   .NOT.ok_share_taxis
     .	   .AND. iaxis.EQ.native_tax
     .     .AND. reused(iaxis).NE.unspecified_int4 ) THEN
	      reused(iaxis) = unspecified_int4
	      dup_guts = .FALSE.
	  ENDIF 
	  IF ( dup_name .AND. .NOT.dup_guts ) THEN
	     line_fixname(iaxis) = .TRUE.  ! new definition reuses old name
	  ELSE
	     line_fixname(iaxis) = .FALSE.
	  ENDIF
	GOTO 100  ! like  100   CONTINUE

* check that all the grids point to real axes
 110	igrid = 0		! to get start of linked list
 250	CALL TM_NEXT_TMP_GRID( igrid, *260)
	   count = tmp_nam_ptr(igrid)
           DO 240 idim = 1, nferdims 
              vname = temp_axnams(idim, count)
!              IF ( vname .EQ. 'RECORD' ) THEN
!* ... the CDF term "RECORD" is the last dim'ed axis (like "UNKNOWN" for GT)
!                 grid_line(idim, igrid) = mpsunkn
              IF ( vname .EQ. 'NORMAL' ) THEN
                 grid_line(idim, igrid) = mpsnorm
              ELSE
* ... first choose from the axis names just defined
	         iaxis = 0	! to get start of linked list
 200	         CALL TM_NEXT_TMP_LINE( iaxis, *202)
                    IF ( vname .EQ. line_name(iaxis) ) THEN
	               IF (reused(iaxis) .EQ. unspecified_int4 ) THEN
                          grid_line(idim, igrid) = iaxis
	               ELSE
                          grid_line(idim, igrid) = reused(iaxis)
	               ENDIF
                       GOTO 240
                    ENDIF
	         GOTO 200
* ... not found? then this name was a duplicate definition. Reuse older defn
 202		 grid_line(idim, igrid) = reused(iaxis)
	         IF (reused(iaxis) .EQ. unspecified_int4)
     .						STOP 'cd_clean_crptn'
              ENDIF
 240       CONTINUE
	GOTO 250
                 
* check for duplicate grid name / guts
 260	igrid = 0		! to get start of linked list
 300	CALL TM_NEXT_TMP_GRID( igrid, *301)

* ... eliminate this grid name to look for duplicates
           vname = grid_name(igrid)   ! temporary storage
           grid_name(igrid) = char_init
           i_dup = TM_GET_GRIDNUM( vname )
           dup_name = i_dup .NE. unspecified_int4

           IF ( dup_name ) THEN
              dup_guts = TM_SAME_GRID_DEF( i_dup, igrid )
           ELSE
              dup_guts = .FALSE. ! ... well not necessarily but who cares ?
           ENDIF

* Save the grid if it's unique
          IF ( .NOT.dup_name ) THEN
             grid_name(igrid) = vname

* Throw away this duplicate but make sure variables point to the original
          ELSEIF ( dup_guts ) THEN
             grid_name(igrid) = char_init   ! throw it away
             DO 290 ivar = 1, maxvars
 290         IF ( ds_grid_number(ivar) .EQ. igrid )
     .               ds_grid_number(ivar) = i_dup

* Duplicate name - differing guts / make the grid name unique
          ELSE
             grid_name(igrid) = vname
             CALL TM_NEW_GRID_NAME( grid_name(igrid), vname )
             grid_name(igrid) = vname
          ENDIF

	GOTO 300	! like 300    CONTINUE

* destroy any axis names that are duplicates both in name and defn
* make unique any axis names that are duplicates but with new guts

* If there is a new name, save the original axis name of the axis as an
* attribute of the coordinate variable in the linked-list attribute structure,
* and change the axis name itself.

*** Here if the name is a dimension not a coordinate variable in the original
*** dataset, we want to do a CD_RENAME_DIM, so that we rename in the dset structure 
*** the dimension but not (for example) the 2-D variable with the same name.

 301	iaxis = 0		! to get start of linked list
 400	CALL TM_NEXT_TMP_LINE( iaxis, *402)
          vname = line_name(iaxis)
          len = TM_LENSTR1(vname)
          CALL CD_GET_VAR_ID (dset, vname, varid, status)

          IF (status .EQ. merr_ok) CALL CD_GET_VAR_INFO (dset, varid, 
     .           vname, vartype, nvdims, vdims, nvatts, coordvar, 
     .            outflag, status)

	  IF ( reused(iaxis) .NE. unspecified_int4 ) THEN
            line_name(iaxis) = char_init16   ! destroy new duplicate defn
            line_fixname(iaxis) = .FALSE.  
            
            buff = line_name(reused(iaxis))

            IF (status .NE. merr_ok) THEN   ! If it's a Dimension, rename it

               CALL CD_GET_DIM_ID (dset, vname, dimid)
               CALL CD_RENAME_DIM (dset, dimid, buff, status)

	    ELSE
               IF (nvdims.EQ.1) CALL CD_RENAME_VAR (dset, varid, buff, status)
               
* Replace or add the orig_file_axname attribute
               attname = 'orig_file_axname'
               CALL CD_GET_VAR_ATT_ID (dset, varid, attname, attid, 
     .              status)
               IF (attid .GT. 0) THEN 
                  CALL CD_GET_VAR_ATT_INFO (dset, varid, attid, attname,
     .               attype, attlen, attoutflag, status)
                  CALL CD_REPLACE_ATTR (dset, varid, attname, NF_CHAR, 
     .               len, vname, dummy, status)
               ELSE
                  CALL CD_PUT_NEW_ATTR (dset, varid, 'orig_file_axname',
     .               NF_CHAR, len, -1, vname, dummy, status)
               ENDIF
            ENDIF

          ELSEIF ( line_fixname(iaxis) ) THEN

            CALL TM_NEW_LINE_NAME( line_name(iaxis), buff )

            line_name(iaxis) = buff

            IF (status .EQ. merr_ok .AND. 
     .          STR_SAME(vname, buff) .NE. 0) THEN

               CALL CD_GET_DIM_ID (dset, vname, dimid)
               CALL CD_RENAME_DIM (dset, dimid, buff, status)
	       
               IF (nvdims.EQ.1) CALL CD_RENAME_VAR (dset, varid, buff, status)

* Replace or add the orig_file_axname attribute
               attname = 'orig_file_axname'
               CALL CD_GET_VAR_ATT_ID (dset, varid, attname, attid, 
     .              status)
               IF (attid .GT. 0) THEN 
                  CALL CD_GET_VAR_ATT_INFO (dset, varid, attid, attname,
     .               attype, attlen, attoutflag, status)
                  CALL CD_REPLACE_ATTR (dset, varid, attname, NF_CHAR, 
     .               len, vname, dummy, status)
               ELSE
                  CALL CD_PUT_NEW_ATTR (dset, varid, 'orig_file_axname',
     .                NF_CHAR, len, -1, vname, dummy, status)
               ENDIF
            ENDIF

          ENDIF
	GOTO 400	! like  400    CONTINUE

* determine the "record" axis
 402	IF ( ds_time_axis(dset) .LT. 1 ) THEN
             ds_time_axis(dset) = unspecified_int4
        ELSE
           CALL CD_GET_DS_DIMS( dset, ds_time_axis(dset), vname, 
     .                          npts, istat )
           ds_time_axis(dset) = TM_GET_LINENUM( vname )

        ENDIF

* For DSG dataset, create a coordinate translation grid
	IF (dsg_ragged(dset)) THEN
* ... allocate a temp grid
*     it triggers grid finalization & use counts in TM_GARB_COL_GRIDS
	   CALL TM_ALLO_TMP_GRID(igrid, status)
	   IF (status .NE. merr_ok) GOTO 5900
           grid_name(igrid) = 'GDSG_COORDS'
           CALL TM_NEW_GRID_NAME( grid_name(igrid), vname )
           grid_name(igrid) = vname
* ... all of the coordinate variables of this DSG are included
	   DO idim = 1, 4
	      iaxis = ds_basic_axes(idim, dset)
	      IF (iaxis .NE. int4_init) THEN
	         grid_line(idim, igrid) = iaxis
	      ELSE
	         grid_line(idim, igrid) = mpsnorm
	      ENDIF	
	   ENDDO
	   grid_line(E_dim, igrid) = dsg_instance_axis
	   grid_line(F_dim, igrid) = mpsnorm
	   IF (dsg_instnc_ftrset_axis .NE. int4_init) 
     .		grid_line(F_dim, igrid) = dsg_instnc_ftrset_axis

* ... save pointer to this grid as part of the dataset
	   dsg_xlate_grid(dset) = igrid
	ENDIF

* successful completion
 	status = merr_ok
        RETURN

* error messages
 5100	buff = 'axis: '//vname//' grid: '//grid_name(igrid)
 	CALL TM_ERRMSG (merr_lineundef, status, 'CD_CLEAN_GRIDS',
     .			no_descfile, no_stepfile,
     .			buff, no_errstring, *5900)

* error exit
 5900   RETURN
	END
