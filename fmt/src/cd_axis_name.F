	CHARACTER*(*) FUNCTION CD_AXIS_NAME ( grid, idim, nlen )

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
* determine the netCDF name to use for the indicated axis of this grid

* Programmer Steve Hankin
* NOAA/PMEL, Seattle, WA - Tropical Modeling and Analysis Program

* revision 0.0 - 6/92
* V4.20  11/95 - remove parentheses surrounding dynamic axis names

* argument definitions:
*     grid    - TMAP grid id
*     idim    - axis orientation (1-4)
*     nlen    - length of name

* include files
#ifdef unix
	include 'tmap_format/tmap_errors.parm'
	include 'tmap_format/tmap_dims.parm'
#include "tmap_format/tmap_dset.parm"
	include 'tmap_format/gt_lib.parm'
	include 'tmap_format/xtm_grid.cmn_text'
	external xgt_grid_data
#else
	INCLUDE 'TMAP_FORMAT:TMAP_DIMS.PARM'
	INCLUDE 'TMAP_FORMAT:TMAP_DSET.PARM'
	INCLUDE 'TMAP_FORMAT:GT_LIB.PARM'
	INCLUDE 'TMAP_FORMAT:XTM_GRID.CMN'
#endif

* argument declarations
      INTEGER grid, idim, nlen

* local variable declarations:
      LOGICAL TM_ABSTRACT_AXIS
      INTEGER TM_LENSTR1, iaxis

      CHARACTER*1 ww_dim_name(4)
      DATA ww_dim_name/'X', 'Y', 'Z', 'T'/

* initialize
      iaxis = grid_line(idim, grid)

* basic name straight from the grid
      CD_AXIS_NAME = line_name(iaxis)
      nlen = TM_LENSTR1( CD_AXIS_NAME )

* if it's an abstract axis make a special name
      IF ( TM_ABSTRACT_AXIS( iaxis ) ) THEN
         CD_AXIS_NAME = ww_dim_name(idim)//'AX'
         nlen = 3

* if it's a dynamic axis remove the parentheses
      ELSEIF ( line_name(iaxis)(1:1) .EQ. '(' ) THEN
	 IF ( nlen .LE. 2) nlen = 3			! should never happen
         CD_AXIS_NAME = line_name(iaxis)(2:nlen-1)
	 nlen = nlen - 2
      ENDIF

      RETURN
      END
