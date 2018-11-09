	SUBROUTINE T0_ADJUST_INIT ( cal_id,
     .				    tunit_id_src, t0_src,
     .				    tunit_id_dst, t0_dst, status )

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
* This routine exists to optimize performance in converting between time axes
* Initialize COMMON in preparation for time coordinate conversion calls to
* T0_ADJUST (see detailed docs in T0_ADJUST.F)

* based upon code from routine TDEST_WORLD_INIT.F
* in this routine, however, no syntax error checking of units and T0 is done

* programmer - steve hankin
* NOAA/PMEL, Seattle, WA - Tropical Modeling and Analysis Program

* V695+ *sh* 11/15 

        IMPLICIT NONE
	include 'tmap_dims.parm'
	include 'xtm_grid.cmn_text'
	external xgt_grid_data
	include 'xunits.cmn_text'
	external xunits_data
	include 'ferret.parm'
	include 'errmsg.parm'
	include 'xregrid.cmn'

* calling arguments
*   cal_id       (input) - calendar id for both src and dst axes (must match)
*   tunit_id_src (input) - time unit id on source axis
*   t0_src       (input) - time origin string on source axis
*   tunit_id_dst (input) - time unit id on destination axis
*   t0_dst       (input) - time origin string on destination axis
*   status       (output) - return status

* calling argument declarations:
	INTEGER cal_id, tunit_id_src, tunit_id_dst, status
	CHARACTER*(*)	t0_src, t0_dst

* internal variable declarations:
	LOGICAL TM_DATE_OK
	INTEGER	TM_GET_CALENDAR_ID ,
     .          src_line, dst_line, src_cal_id, dst_cal_id
	REAL	CAL_FACTR
	REAL*8  SECS_FROM_BC, at0src, at0dst, usrc, udst, factor

* initialize
	cnvrt_src_grid = unspecified_int4

* determine the units and absolute T0 for the two axes
	usrc = un_convert( tunit_id_src )
	udst = un_convert( tunit_id_dst )

	at0src = SECS_FROM_BC( t0_src, cal_id, status )
	IF ( status .NE. ferr_ok ) GOTO 5000
	at0dst = SECS_FROM_BC( t0_dst, cal_id, status )
	IF ( status .NE. ferr_ok ) GOTO 5000

* determine the slope and intercept to be used in TDST_WORLD regridding
	tcnvrt_slope = usrc/udst
	tcnvrt_intercept = (at0src-at0dst) / udst

* success
	tcnvrt_ok = .TRUE.
	RETURN

* error exit
 5000	tcnvrt_ok = .FALSE.
	RETURN
	END




