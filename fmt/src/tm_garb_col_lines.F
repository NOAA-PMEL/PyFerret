	SUBROUTINE TM_GARB_COL_LINES ( first, last )

*
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
*
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
	include 'tmap_errors.parm'
	include 'tmap_dims.parm'
#include "tmap_dset.parm"
#include "gt_lib.parm"
	include 'xtm_grid.cmn_text'
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
