	SUBROUTINE CD_CLEAN_GRIDS ( dset, cdfid, temp_axnams, status )


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

* argument definitions:
*       dset    - pointer to TMAP data set
*	cdfid	- netCDF id number for already opened CDF file
*       temp_axnams - temporary storage of axis names to construct grids
*       temp_grid_ptr - points from temporary name storage to actual grid
*	status	- return status

* include files
#ifdef unix
	include 'tmap_errors.parm'
	include 'tmap_dims.parm'
#include "tmap_dset.parm"
#include "gt_lib.parm"
        include 'xbuild_grids.cmn'
	include 'xtm_grid.cmn_text'
	external xgt_grid_data
        include 'xdset_info.cmn_text'
#else
        INCLUDE 'TMAP_FORMAT:XBUILD_GRIDS.CMN'
	INCLUDE 'TMAP_FORMAT:TMAP_ERRORS.PARM'
	INCLUDE 'TMAP_FORMAT:TMAP_DIMS.PARM'
	INCLUDE 'TMAP_FORMAT:TMAP_DSET.PARM'
	INCLUDE 'TMAP_FORMAT:GT_LIB.PARM'
	INCLUDE 'TMAP_FORMAT:XTM_GRID.CMN'
        INCLUDE 'TMAP_FORMAT:XDSET_INFO.CMN_TEXT'
#endif

* argument declarations
	INTEGER	 dset, cdfid, status
*4/99 *kob*
        CHARACTER*64 temp_axnams(4,max_temp_grid)

* local parameter definitions:
      INTEGER     max_grd_per_cdf
      PARAMETER ( max_grd_per_cdf = 40 )

* local variable declarations:
      LOGICAL TM_SAME_GRID_DEF,
     .        dup_name, dup_guts, fix_axnam(max_lines:line_ceiling)
      INTEGER  TM_GET_LINEMATCH,  TM_GET_LINENUM, TM_GET_GRIDNUM,
     .        igrid, iaxis, count, idim, i_dup, ivar,
     .        npts, cdfstat,  reused(max_lines:line_ceiling)
* upped vname to 64 - also removed the *24 from the CHARACTER
      CHARACTER vname*64, buff*64

***** AXES ******
* check for duplicate name / guts of axes
* if completely identical (name and guts) then eliminate the new one
* if new definition with already used name then flag this
	iaxis = 0	! to get start of linked list
 100	CALL TM_NEXT_TMP_LINE( iaxis, *110)
	  reused(iaxis) = TM_GET_LINEMATCH( iaxis, dup_name, dup_guts )
	  IF ( dup_name .AND. .NOT.dup_guts ) THEN
	     fix_axnam(iaxis) = .TRUE.	  ! new definition reuses old name
	  ELSE
	     fix_axnam(iaxis) = .FALSE.
	  ENDIF
	GOTO 100  ! like  100   CONTINUE

* check that all the grids point to real axes
 110	igrid = 0		! to get start of linked list
 250	CALL TM_NEXT_TMP_GRID( igrid, *260)
	   count = tmp_nam_ptr(igrid)
           DO 240 idim = 1, 4 
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
 301	iaxis = 0		! to get start of linked list
 400	CALL TM_NEXT_TMP_LINE( iaxis, *402)
	  IF ( reused(iaxis) .NE. unspecified_int4 ) THEN
	    line_name(iaxis) = char_init16   ! destroy new duplicate defn
          ELSEIF ( fix_axnam(iaxis) ) THEN
            CALL TM_NEW_LINE_NAME( line_name(iaxis), buff )
            line_name(iaxis) = buff
          ENDIF
	GOTO 400	! like  400    CONTINUE

* determine the "record" axis
 402	IF ( ds_time_axis(dset) .LT. 1 ) THEN
             ds_time_axis(dset) = unspecified_int4
        ELSE
           CALL NCDINQ( cdfid,ds_time_axis(dset),vname,npts,cdfstat )
           ds_time_axis(dset) = TM_GET_LINENUM( vname )
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
