	SUBROUTINE TM_MAKE_BASIC_AXIS( name, start,
     .                                 delta, npts, iaxis, status )

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
* Create a very basic 1 to n axis with no frills

* Programmer Steve Hankin
* NOAA/PMEL, Seattle, WA - Tropical Modeling and Analysis Program

* revision 0.0 - 2/92
* V510 *sh* 3/00 - convert to dynamic grids and axes for netCDF files
* V550 *sh* 11/02 - initialize line_modulo_len (just for good house keeping)
* V581 *acm* 6/05 - For fix to bug 1271, add flag line_shift_origin
* V63  *acm* 9/09 - Fix bug 1686: set line_name_orig
* v694 1/15 *acm* for ticket 2227: Set the new line_dim_only flag. If a 
*                 dimension from a nc file is not also a 1-D coordinate 
*                 var, don't write the axis Ferret creates.

*             ACM 12/10/2020
*             bug fix - Ferret issue #1984.  The setting for line_dsg_dset was
*             not initialized (may not be a bug but good to be complete).

* argument definitions:
*       name      - name for axis
*       start     - starting value
*	delta     - delta value
*       npts      - number of points
*       iaxis     - returned axis pointer in TMAP common
*	status	  - return status

* include files
	include 'tmap_errors.parm'
	include 'tmap_dims.parm'
#include "tmap_dset.parm"
#include "gt_lib.parm"
#include "implicit.parm"
	include 'xdset_info.cmn_text'
	external xdset_info_data
	include 'xtm_grid.cmn_text'
	external xgt_grid_data


* argument declarations
	INTEGER	 npts, iaxis, status
        REAL*8   start, delta
        CHARACTER*(*) name

* local variable declarations:
      INTEGER STR_UPCASE,
     .        istat

* allocate a temporary slot for the axis
	CALL TM_ALLO_TMP_LINE(iaxis, status)
	IF (status .NE. merr_ok) GOTO 5900

* build axis
      istat = STR_UPCASE( line_name(iaxis), name )
      line_name_orig(iaxis) = name
      line_dim( iaxis)      = npts
      line_start(iaxis)     = start
      line_delta(iaxis)     = delta
      line_regular(iaxis)   = .TRUE.
      line_units(iaxis)     = ' '
      line_unit_code(iaxis) = 0
      line_modulo(iaxis)    = .FALSE.
      line_modulo_len(iaxis)= 0.0D0
      line_t0(iaxis)        = char_init20
      line_shift_origin(iaxis) = .FALSE.
      line_tunit(iaxis)     = real4_init
      line_direction(iaxis) = 'NA'
      line_dim_only(iaxis)   = .TRUE.
      line_dsg_dset(iaxis)   = pline_not_dsg

* successful completion
      status = merr_ok

 5900 RETURN

	END
