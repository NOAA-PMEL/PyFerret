	SUBROUTINE TM_GARB_COL_LINES ( first, last )

* garbage collect wasted space in newly defined lines (axes)

* Programmer Steve Hankin
* NOAA/PMEL, Seattle, WA - Tropical Modeling and Analysis Program

* revision 0.0 - 2/92
*
* IBM PORT 14apr94 *kob* : Port to IBM revealed a transfer controll problem at 
*			   around label 200.  A "goto" statment was passing control to
*			   a continue statment.
* argument definitions:
*       first   - highest number line to check
*       last    - lowest number line to check
* note: lines are allocated from high to low

* include files
#ifdef unix
	include 'tmap_format/tmap_errors.parm'
	include 'tmap_format/tmap_dims.parm'
#include "tmap_format/tmap_dset.parm"
	include 'tmap_format/gt_lib.parm'
	include 'tmap_format/xtm_grid.cmn_text'
	external xgt_grid_data
#else
	INCLUDE 'TMAP_FORMAT:TMAP_ERRORS.PARM'
	INCLUDE 'TMAP_FORMAT:TMAP_DIMS.PARM'
	INCLUDE 'TMAP_FORMAT:TMAP_DSET.PARM'
	INCLUDE 'TMAP_FORMAT:GT_LIB.PARM'
	INCLUDE 'TMAP_FORMAT:XTM_GRID.CMN'
#endif

* argument declarations
	INTEGER	 first, last

* local variable declarations:
      INTEGER iaxis, hiaxis, igrid, i, idim, next_mem, freed_mem

* there is no way that grids can be using unnamed axes
* delete 'em incrementing the grid axis pointers and compacting line memory

* initialize axis loop
!  like: DO 400 iaxis = first, last   but hiaxis keeps shifting down
      iaxis = first
      hiaxis = last

* is this a hole ?
 10     IF ( line_name(iaxis) .NE. char_init ) GOTO 400

* modify axis pointer saved in the grids
           DO 110 igrid = 1, max_grids
              IF ( grid_name(igrid) .EQ. char_init ) GOTO 110
              DO 100 idim = 1, 4
                 IF ( grid_line(idim, igrid) .EQ. iaxis )
     .         write (6,*) 'grid defns are garbled!: TM_GRID_GARBAGE' ! diag
                 IF ( grid_line(idim, igrid) .GT. iaxis
     .          .AND. grid_line(idim, igrid) .LE. hiaxis )
     .                grid_line(idim,igrid) = grid_line(idim,igrid) - 1
 100          CONTINUE
 110       CONTINUE

* is there line memory to free ?
* this next goto was 200, but the IBM compiler wouldn't accept a 
* transfer of control to a continue statment.  *kob* 14apr94
           IF ( line_regular(iaxis) ) GOTO 205

* look ahead to the next line with uneven point spacing 
           DO 200 i = iaxis+1, hiaxis
              IF ( .NOT.line_regular(i) ) THEN
                 next_mem = line_subsc1(i)
                 GOTO 210
              ENDIF
 200       CONTINUE
 205       next_mem  = next_line_mem_pos
 210       freed_mem = next_mem - line_subsc1(iaxis)

* shift this much line memory to fill the hole
           DO 250 i = next_mem, next_line_mem_pos
 250       line_mem(i-freed_mem) = line_mem(i)

* update line memory pointers
           DO 280 i = iaxis+1, hiaxis
              IF ( .NOT.line_regular(i) ) THEN
                 line_subsc1(i) = line_subsc1(i) - freed_mem
              ENDIF
 280       CONTINUE
           next_line_mem_pos = next_line_mem_pos - freed_mem
           
* shift all the definitions back to fill the hole
           DO 300 i = iaxis+1, hiaxis
 300       CALL TM_COPY_LINE( i, i-1 )

* initialize new hole at end
           line_name(hiaxis) = char_init
           line_subsc1(hiaxis) = unspecified_int4
           hiaxis = hiaxis - 1
           iaxis = iaxis - 1   ! nullify decrement below

! like: 400    CONTINUE
 400    iaxis = iaxis + 1
        IF ( iaxis .LE. hiaxis ) GOTO 10

* successful completion
        RETURN

	END
