	LOGICAL FUNCTION GEOG_COS_FACTOR( idim, grid )

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
* determine whether a particular axis of a grid represents geographical
* locations (lat.,long., etc.) requiring COS corrections

* programmer - steve hankin
* NOAA/PMEL, Seattle, WA - Tropical Modeling and Analysis Program
* written for VAX computer under VMS operating system
*
* revision 0.0 - 8/25/88
* V230:  7/23/92 - TRUE only if grid has a valid latitude axis (major rewrite)

#ifdef unix
	include 'tmap_format/tmap_dims.parm'
	include 'tmap_format/xtm_grid.cmn_text'
	external xgt_grid_data
        include 'tmap_format/xunits.cmn_text'
        external xunits_data
	include 'ferret_cmn/ferret.parm'
	include 'ferret_cmn/xtext_info.cmn'
#else
	INCLUDE 'TMAP_FORMAT:TMAP_DIMS.PARM'
	INCLUDE 'TMAP_FORMAT:XTM_GRID.CMN'
        INCLUDE 'TMAP_FORMAT:XUNITS.CMN'
	INCLUDE 'FERRET_CMN:FERRET.PARM'
	INCLUDE 'FERRET_CMN:XTEXT_INFO.CMN'
#endif

* calling argument declarations:
	INTEGER		idim, grid

* internal variable declarations:
	INTEGER		xaxis, yaxis

* check for program bug	! temp
	IF ( grid .EQ. unspecified_int4 ) STOP 'no_grd_orient'

* initialize
	xaxis = grid_line( x_dim, grid )
	yaxis = grid_line( y_dim, grid )

* check that the X and Y axes of this grid are both degrees
* assume the calling routine calls this only for lat or long
*     (so the orientation of idim isn't checked here)

	IF ( xaxis .EQ. mnormal .OR. xaxis .EQ. munknown
     .  .OR. yaxis .EQ. mnormal .OR. yaxis .EQ. munknown ) THEN
	   GEOG_COS_FACTOR = .FALSE.
	ELSE
           GEOG_COS_FACTOR = line_unit_code(xaxis) .EQ. pun_degrees
     .                 .AND. line_unit_code(yaxis) .EQ. pun_degrees
	ENDIF

	RETURN
	END
