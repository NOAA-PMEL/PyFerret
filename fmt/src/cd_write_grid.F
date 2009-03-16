	SUBROUTINE CD_WRITE_GRID ( cdfid, dset, grid, lo, hi, recdim, 
     .                           itsa_slab, gname, edges_flag, 
     .                           do_bounds, mode_upcase_output, 
     .                           keepax_flag, status )

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
* Write a grid definition out to the current output netCDF file

* Programmer Steve Hankin
* NOAA/PMEL, Seattle, WA - Tropical Modeling and Analysis Program

* revision 0.0 - 3/92
*               10/92 - put into netCDF into define mode before writing attribs
*		1/95  - change in "parent" argument to cd_write_axis (now int)
*   11/95 - *sh* use CD_AXIS_NAME rather than line_name for dynamic lines
*    2/96 - *sh* same deal for dynamic grids ... a quick fix right here
* v500 *kob*  3/99 - change lengths of CD_CHILDAX_NAME, CD_AXIS_NAME,
*                    lname to 64 from 24 to match longer line_name
* v570 *acm*  6/04 - changes to use bounds attribute
* V602 *acm* 11/06 - Fix bug 1471; new routine CD_AXIS_OUTNAME gets correct 
*                    output axis name.
* V62  *acm*  2/09 - Fix bug 1634: allow SAVE/KEEP_AXISNAMES so that child axis names
*                    arent written. They can get too long for OPeNDAP library calls

* argument definitions:
*     cdfid   - netCDF id number for already opened CDF file
*     grid    - TMAP grid id
*     lo      - low indices of variable hyperslab
*     hi      - high indices of variable hyperslab
*     recdim  - which axis (1-4) is to be the CDF record axis
*     itsa_slab- does the underlying variable need a formal grid defn
*     gname   - the name of the parent grid created
*     status  - return status

* include files
        include 'netcdf.decl'
#include "netcdf.inc"
	include 'tmap_errors.parm'
	include 'tmap_dims.parm'
        include 'cdf_tmap.parm'
#include "gt_lib.parm"
        include 'cd_lib.parm'
	include 'xtm_grid.cmn_text'
	external xgt_grid_data


* argument declarations
      LOGICAL itsa_slab, do_bounds, mode_upcase_output
      INTEGER cdfid, dset, grid, lo(4), hi(4), recdim, edges_flag, status
      CHARACTER*(*) gname

* local variable declarations:
      LOGICAL gmod
      INTEGER TM_LENSTR1, 
     .        defid, defdim, nlen, slen, cdfstat, iaxis, idim, glen, i
* *kob*  3/99 - change lengths of CD_CHILDAX_NAME, CD_AXIS_NAME,
*                    lname to 64 from 24 to match longer line_name
      CHARACTER CD_CHILDAX_NAME*64, CD_AXIS_OUTNAME*64,
     .		names*132, lname*64
      CHARACTER*1 ww_dim_name(4)
      DATA ww_dim_name/'X', 'Y', 'Z', 'T'/

* are the axes, alone, enough of a grid definition
      IF ( .NOT.itsa_slab ) GOTO 100

* put CDF into define mode so attributes can be written
      CALL CD_SET_MODE(cdfid, pcd_mode_define, status)
      IF ( status .NE. merr_ok ) RETURN

* create "grid_definition" dimension name if not already in the file
      defdim = NCDID( cdfid, 'grid_definition', cdfstat )
      IF ( cdfstat .NE. NCNOERR ) defdim =
     .     NCDDEF( cdfid, 'grid_definition', 1, cdfstat )

* create an output grid name that reflects axes which have been averaged, etc.
* and a string with a list of the axis names (starting w/ 2 blanks)
      gname = grid_name(grid)
      glen = TM_LENSTR1( gname )
! 2/96 - strip parentheses from dynamic grid name
      IF ( gname(1:1).EQ.'(' .AND. glen.GT.2 ) THEN
	gname = gname(2:glen-1)
	glen = glen - 2
      ENDIF
! 2/96 - end of insertion
      gmod = .FALSE.
      names = ' '
      nlen = 1
      DO 50 idim = 1, 4
         iaxis = grid_line(idim, grid)

         lname = CD_AXIS_OUTNAME(dset, grid, idim, slen)
         IF ( iaxis .EQ. mpsnorm ) THEN
            names = names(:nlen)//' '//'NORMAL'
         ELSEIF ( lo(idim) .EQ. unspecified_int4 ) THEN
            names = names(:nlen)//' '//'NORMAL'
            IF ( gmod ) THEN
               gname = gname(:glen)//ww_dim_name(idim)
               glen = glen + 1
            ELSE
               gname = gname(:glen)//'_N'//ww_dim_name(idim)
               gmod = .TRUE.
               glen = glen + 3
            ENDIF

         ELSEIF ( lname .EQ. 'ABSTRACT' .AND. (keepax_flag.EQ.0)) THEN
            names = names(:nlen)//' '//
     .           CD_CHILDAX_NAME( ww_dim_name(idim)//'AX',
     .                            lo(idim), hi(idim), i )
         ELSE
            names = names(:nlen)//' '//lname
         ENDIF
         nlen = TM_LENSTR1(names)
 50   CONTINUE

* does this grid already exist ? (no consistency check if so)
      defid = NCVID( cdfid, gname(:glen), cdfstat )
      IF ( cdfstat .EQ. NCNOERR ) GOTO 1000

* create a dummy variable for this grid definition
      defid = NCVDEF( cdfid, gname(:glen), NCCHAR, 1,
     .                defdim, cdfstat )

* write the axis list as an attribute of the grid
      CALL NCAPTC( cdfid, defid, 'axes', NCCHAR, nlen-2,
     .             names(3:), cdfstat )

* create the parent axes
 100  DO 200 idim = 1, 4
         IF ( grid_line(idim, grid) .NE. mpsnorm
     .  .AND. lo(idim) .NE. unspecified_int4     ) THEN
            CALL CD_WRITE_AXIS( cdfid, dset, grid, idim, 
     .           idim.EQ.recdim,
     .           lo(idim), hi(idim), pcdf_full_span, 
     .           edges_flag, do_bounds, mode_upcase_output,
     .           keepax_flag, status ) 

            IF ( status .NE. merr_ok ) GOTO 5900
         ENDIF
 200  CONTINUE

* successful completion
 1000 status = merr_ok
      RETURN

* error exit
 5900   RETURN
	END
