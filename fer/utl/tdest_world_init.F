	SUBROUTINE TDEST_WORLD_INIT ( src_grid, dst_grid, status )

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
* Initialize COMMON in preparation for time coordinate conversion calls to
* TDEST_WORLD (see detailed docs in tdest_world.F)

* This routine exists to optimize performance in converting between time axes

* programmer - steve hankin
* NOAA/PMEL, Seattle, WA - Tropical Modeling and Analysis Program

* V312:  6/2/94

#ifdef unix
	include 'tmap_format/tmap_dims.parm'
	include 'tmap_format/xtm_grid.cmn_text'
	external xgt_grid_data
	include 'tmap_format/xunits.cmn_text'
	external xunits_data
	include 'ferret_cmn/ferret.parm'
	include 'ferret_cmn/errmsg.parm'
	include 'ferret_cmn/xregrid.cmn'
#else
	INCLUDE 'TMAP_FORMAT:TMAP_DIMS.PARM'
	INCLUDE 'TMAP_FORMAT:XTM_GRID.CMN'
	INCLUDE 'TMAP_FORMAT:XUNITS.CMN_TEXT'
	INCLUDE 'FERRET_CMN:FERRET.PARM'
	INCLUDE 'FERRET_CMN:ERRMSG.PARM'
	INCLUDE 'FERRET_CMN:REGRID.CMN'
#endif

* calling argument declarations:
	INTEGER	src_grid, dst_grid, status

* internal variable declarations:
	LOGICAL TM_DATE_OK
	INTEGER	src_line, dst_line
	REAL*4	usrc, udst
	REAL*8  SECS_FROM_BC, at0src, at0dst

* initialize
	tcnvrt_ok = .FALSE.
	cnvrt_src_grid = src_grid
	status = ferr_ok

* first check to see if calendar-based regridding between these two grids
* is possible
	IF ( src_grid .EQ. unspecified_int4
     .	.OR. dst_grid .EQ. unspecified_int4 ) STOP 'TDEST_WORLD_INIT'
	src_line = grid_line( t_dim, src_grid )
	dst_line = grid_line( t_dim, dst_grid )
	IF ( src_line .EQ. mnormal .OR. src_line .EQ. munknown
     .	.OR. dst_line .EQ. mnormal .OR. dst_line .EQ. munknown )
     .		CALL ERRMSG( ferr_internal, status,
     .				'date regrid on missing T axis', *5000 )
	IF ( line_direction(src_line) .NE. 'TI'
     .	.OR. line_direction(dst_line) .NE. 'TI' ) RETURN
	IF ( .NOT. (TM_DATE_OK(line_t0(src_line)) 
     .	       .AND.TM_DATE_OK(line_t0(dst_line))) ) RETURN

* determine the units and absolute T0 for the two axes
	usrc = un_convert( line_unit_code(src_line) )
	udst = un_convert( line_unit_code(dst_line) )
	at0src = SECS_FROM_BC( line_t0(src_line), status )
	IF ( status .NE. ferr_ok ) RETURN
	at0dst = SECS_FROM_BC( line_t0(dst_line), status )
	IF ( status .NE. ferr_ok ) RETURN

* determine the slope and intercept to be used in TDST_WORLD regridding
	tcnvrt_slope = usrc/udst
	tcnvrt_intercept = (at0src-at0dst) / udst

* success
	tcnvrt_ok = .TRUE.
	RETURN

* error exit
 5000	RETURN
	END




