	SUBROUTINE CD_CLEAN_GRIDS ( dset, cdfid, temp_axnams, status )

* check for errors and eliminate duplication from newly defined grids and axes
* When the routine exits there will be no duplicate axis names although there
* will be holes in line_memory from deleted axes.
* When the routine exits there will still be duplicate grid names among the
* newly defined grids.  These may be eliminated after variables are associated.
* Finally, determine the "record" axis of this data set,

* Programmer Steve Hankin
* NOAA/PMEL, Seattle, WA - Tropical Modeling and Analysis Program

* revision 0.0 - 2/92

* argument definitions:
*       dset    - pointer to TMAP data set
*	cdfid	- netCDF id number for already opened CDF file
*       temp_axnams - temporary storage of axis names to construct grids
*       temp_grid_ptr - points from temporary name storage to actual grid
*	status	- return status

* include files
#ifdef unix
        include 'tmap_format/xbuild_grids.cmn'
	include 'tmap_format/tmap_errors.parm'
	include 'tmap_format/tmap_dims.parm'
#include "tmap_format/tmap_dset.parm"
	include 'tmap_format/gt_lib.parm'
	include 'tmap_format/xtm_grid.cmn_text'
	external xgt_grid_data
        include 'tmap_format/xdset_info.cmn_text'
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
        CHARACTER*16 temp_axnams(4,max_temp_grid)

* local parameter definitions:
      INTEGER     max_grd_per_cdf
      PARAMETER ( max_grd_per_cdf = 40 )

* local variable declarations:
      LOGICAL TM_SAME_LINE_DEF, TM_SAME_GRID_DEF,
     .        dup_name, dup_guts, fix_axnam( 0:max_lines )
      INTEGER TM_LENSTR1, TM_GET_LINENUM, TM_GET_GRIDNUM,
     .        igrid, iaxis, vlen, count, idim, i_dup_name, ivar,
     .        npts, cdfstat
      CHARACTER*24 vname*24, buff*64

***** AXES ******
* check for duplicate name / guts of axes
* if completely identical (name and guts) then eliminate the new one
* if new definition with already used name then flag this
        DO 100 iaxis = firstax, lastax
* ... temporarily eliminate name to look for duplicates
          vname = line_name(iaxis)
          vlen = TM_LENSTR1( vname )
          line_name(iaxis) = char_init

          i_dup_name = TM_GET_LINENUM( vname(:vlen) )
          dup_name = i_dup_name .NE. unspecified_int4
          IF ( dup_name ) THEN
             dup_guts = TM_SAME_LINE_DEF( i_dup_name, iaxis )
          ELSE
             dup_guts = .FALSE. ! ... well not necessarily but who cares ?
          ENDIF

* Save the axis name and line memory storage if it's unique
          IF ( .NOT.dup_name ) THEN
             line_name(iaxis) = vname
             fix_axnam(iaxis)  = .FALSE.

* Identical repeat of a previous definition - hose this axis
          ELSEIF ( dup_guts ) THEN
             fix_axnam(iaxis)  = .FALSE.

* Duplicate name but  differing guts / save it for now - rename it at the end
          ELSE
             fix_axnam(iaxis)  = .TRUE.
             line_name(iaxis) = vname
          ENDIF
 100   CONTINUE

* check that all the grids point to real axes
        DO 250 igrid = firstgrid, lastgrid
           count = igrid - firstgrid + 1 
           DO 240 idim = 1, 4 
              vname = temp_axnams(idim, count)
!              IF ( vname .EQ. 'RECORD' ) THEN
!* ... the CDF term "RECORD" is the last dim'ed axis (like "UNKNOWN" for GT)
!                 grid_line(idim, igrid) = mpsunkn
              IF ( vname .EQ. 'NORMAL' ) THEN
                 grid_line(idim, igrid) = mpsnorm
              ELSE
* ... first choose from the axis names just defined
                 DO 200 iaxis = firstax, lastax
                    IF ( vname .EQ. line_name(iaxis) ) THEN
                       grid_line(idim, igrid) = iaxis
                       GOTO 240
                    ENDIF
 200             CONTINUE
* ... failing that choose from axis names previously defined
                 DO 210 iaxis = 1, max_lines   ! ok ??
                    IF ( vname .EQ. line_name(iaxis) ) THEN
                       grid_line(idim, igrid) = iaxis
                       GOTO 240
                    ENDIF
 210             CONTINUE
                 GOTO 5100    ! non-existent axis
              ENDIF
 240       CONTINUE
 250    CONTINUE
                 
* check for duplicate grid name / guts
        DO 300 igrid = firstgrid, lastgrid

* ... eliminate this grid name to look for duplicates
           vname = grid_name(igrid)   ! temporary storage
           grid_name(igrid) = char_init
           i_dup_name = TM_GET_GRIDNUM( vname )
           dup_name = i_dup_name .NE. unspecified_int4

           IF ( dup_name ) THEN
              dup_guts = TM_SAME_GRID_DEF( i_dup_name, igrid )
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
     .               ds_grid_number(ivar) = i_dup_name

* Duplicate name - differing guts / make the grid name unique
          ELSE
             grid_name(igrid) = vname
             CALL TM_NEW_GRID_NAME( grid_name(igrid), vname )
             grid_name(igrid) = vname
          ENDIF

 300    CONTINUE

* make unique any axis names that are duplicates but with new guts
        DO 400 iaxis = firstax, lastax
           IF ( fix_axnam(iaxis) ) THEN
              CALL TM_NEW_LINE_NAME( line_name(iaxis), buff )
              line_name(iaxis) = buff
           ENDIF
 400    CONTINUE

* determine the "record" axis
        IF ( ds_time_axis(dset) .LT. 1 ) THEN
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
