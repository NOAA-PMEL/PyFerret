	CHARACTER*(*) FUNCTION CD_AXIS_OUTNAME ( dset, grid, idim, nlen )

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
* determine the netCDF name to use for writing the indicated axis of this grid

* Programmer Ansley Manke
* NOAA/PMEL, Seattle, WA - Tropical Modeling and Analysis Program

* revision 0.0 - 11/06
* V62   2/09 *acm* When writing user-defined variables and when MODE UPCASE_OUTPUT is
*                  cancelled, get the original upper/lowercase spelling of the axis name.
* V63   9/09 *acm* Fixing bug 1686: if line_name_orig is not set, use line_name

* argument definitions:
*     dset    - dset (if netcdf file, for getting name and attribute 
*               info from linked-list structure  
*     grid    - TMAP grid id
*     idim    - axis orientation (1-4)
*     nlen    - length of name

* include files
	include 'tmap_errors.parm'
	include 'tmap_dims.parm'
#include "tmap_dset.parm"
#include "gt_lib.parm"
	include 'xtm_grid.cmn_text'
	external xgt_grid_data
        include 'netcdf.decl'
#include "netcdf.inc"

* argument declarations
      INTEGER dset, grid, idim, nlen

* local variable declarations:
      LOGICAL TM_ABSTRACT_AXIS, NC_GET_ATTRIB, coordvar, got_it
      INTEGER TM_LENSTR1, STR_SAME, iaxis, varid, nvatts, status, 
     .  attlen, maxlen, attoutflag, alen
      REAL val

      CHARACTER*1 ww_dim_name(4)
      CHARACTER varname*130, buff*130, attname*128
      DATA ww_dim_name/'X', 'Y', 'Z', 'T'/

* initialize
      iaxis = grid_line(idim, grid)
      buff = ' '

* basic name straight from the grid

      varname = line_name(iaxis)
      nlen = TM_LENSTR1( varname )

* Get information, if it exists, from linked-list structure, including
* original upper/lowercase form of axis name. Also restore the original
* name if the axis name was changed on inupt. (If name matched an existing 
* line name, then Ferret has changed the name for uniqueness.)

* Do not try to get the axis name if its in parentheses. These names are
* dynamic axis names.

      status = merr_ok + 1
      IF (varname(1:1) .NE. '(') 
     .   CALL CD_GET_VAR_ID (dset, varname, varid, status)  
     
      got_it = .FALSE.
      IF (status .EQ. merr_ok) THEN
         attname = 'orig_file_axname' 
         alen = 18

         maxlen = nf_max_name ! from netcdf.inc
         got_it = NC_GET_ATTRIB(dset, varid, attname(1:alen),
     .                       .FALSE., varname(1:nlen), maxlen, attlen, 
     .                       attoutflag, buff, val)

         IF ( got_it ) THEN
            CD_AXIS_OUTNAME = buff
            nlen = TM_LENSTR1( buff )
         ELSE 
            CD_AXIS_OUTNAME = line_name_orig(iaxis)
	    IF (STR_SAME(line_name_orig(iaxis), char_init) .EQ. 0) 
     .        CD_AXIS_OUTNAME = line_name(iaxis)
            nlen = TM_LENSTR1( CD_AXIS_OUTNAME )
	    IF (nlen .LE. 1  .OR.  varname(1:1) .EQ. '(' ) THEN
               CD_AXIS_OUTNAME = varname
	       nlen = TM_LENSTR1( CD_AXIS_OUTNAME )
	    ENDIF
         ENDIF

      ELSE

         CD_AXIS_OUTNAME = line_name_orig(iaxis)
         IF (STR_SAME(line_name_orig(iaxis), char_init) .EQ. 0) 
     .        CD_AXIS_OUTNAME = line_name(iaxis)
         nlen = TM_LENSTR1( CD_AXIS_OUTNAME )
	 IF (nlen .LE. 1  .OR.  varname(1:1) .EQ. '(' ) THEN
            CD_AXIS_OUTNAME = varname
	    nlen = TM_LENSTR1( CD_AXIS_OUTNAME )
	 ENDIF

      ENDIF

* if it's an abstract axis make a special name
      IF ( TM_ABSTRACT_AXIS( iaxis ) ) THEN
         CD_AXIS_OUTNAME = ww_dim_name(idim)//'AX'
         nlen = 3

* if it's a dynamic axis and if we didnt recover the 
* original name from the linked-list information,
* remove the parentheses

      ELSEIF ( line_name(iaxis)(1:1) .EQ. '(' .AND.
     .         .NOT.got_it ) THEN
	 IF ( nlen .LE. 2) nlen = 3			! should never happen
         CD_AXIS_OUTNAME = line_name(iaxis)(2:nlen-1)
	 nlen = nlen - 2
      ENDIF

      nlen = TM_LENSTR1( CD_AXIS_OUTNAME )

      RETURN
      END
