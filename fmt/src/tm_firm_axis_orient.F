	LOGICAL FUNCTION TM_FIRM_AXIS_ORIENT(axis, orientation)

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
* Determine if "axis" has a rock-firm orientation

* Programmer Steve Hankin
* NOAA/PMEL, Seattle, WA - Tropical Modeling and Analysis Program

* V500 *sh* 6/99 - new -- to help with SET DATA/ORDER=

* argument definitions:
*	axis	    - input:  axis in TMAP COMMON
*	orientation - output: a value 1-4 if orientation is firm

* include files
#ifdef unix
	include 'tmap_dims.parm'
#include "tmap_dset.parm"
#include "gt_lib.parm"
	include 'xtm_grid.cmn_text'
	external xgt_grid_data
	include 'xunits.cmn_text'
	external xunits_data
#else
        INCLUDE 'TMAP_FORMAT:NETCDF.DECL'  ! untested
#include "netcdf.inc"
	INCLUDE 'TMAP_FORMAT:TMAP_DIMS.PARM'
	INCLUDE 'TMAP_FORMAT:TMAP_DSET.PARM'
	INCLUDE 'TMAP_FORMAT:GT_LIB.PARM'
	INCLUDE 'TMAP_FORMAT:XDSET_INFO.CMN'
	INCLUDE 'TMAP_FORMAT:XTM_GRID.CMN'
	INCLUDE 'TMAP_FORMAT:XUNITS.CMN'
#endif

* argument declarations
	INTEGER	 axis, orientation

* local variable declarations:
	LOGICAL TM_HAS_STRING
	INTEGER units

* initialize
	units = line_unit_code(axis)

* try to determine the orientation of the axis

	IF ( units .LT. 0 ) THEN
	  orientation = t_dim
	ELSEIF ( units .EQ. 4 ) THEN     ! degrees
	  IF  ( TM_HAS_STRING(line_units(axis), 'NORTH')
     .     .OR. TM_HAS_STRING(line_units(axis), 'SOUTH')
     .     .OR. TM_HAS_STRING(line_name(axis), 'LAT') ) THEN 
	    orientation = y_dim
	  ELSEIF ( TM_HAS_STRING(line_units(axis), 'EAST')
     .	   .OR. TM_HAS_STRING(line_units(axis), 'WEST')
     .     .OR. TM_HAS_STRING(line_name(axis), 'LON') ) THEN 
            orientation = x_dim
	  ENDIF
	ELSEIF ( units .EQ. pun_millibars
     .	 .OR.  units .EQ. pun_decibars
     .   .OR.  units .EQ. 14
     .   .OR.  units .EQ. 15                  ! layer or level
     .   .OR. TM_HAS_STRING(line_name(axis), 'HEIGHT') 
     .   .OR. TM_HAS_STRING(line_name(axis), 'ELEV') ) THEN 
          orientation = z_dim
	ELSE
	  orientation = 0
	ENDIF

* set return value
	TM_FIRM_AXIS_ORIENT = orientation .NE. 0
  
	RETURN
	END
