	SUBROUTINE GCF_GET_AXIS_SRCS(ifcn, iarg, axis_source)

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
* get merged into the joint context of a grid-changing function

* V450: 1/97
*	7/97 - slight reorg for external functions
*   8/97 - type mismatch on axis_implied_from caught by 
*              linux/aix compilers *kob*

* calling argument declarations
	INTEGER	ifcn, iarg, axis_source(4)

* internal variable declarations
	INTEGER	idim, iptr1,
     .		axis_will_be(4)
	INTEGER efcn_get_num_reqd_args
	LOGICAL axis_implied_from(4)

#ifdef unix
	include 'tmap_format/tmap_dims.parm'
        include 'tmap_format/xtm_grid.cmn_text'
	include 'ferret_cmn/ferret.parm'
	include 'ferret_cmn/xgrid_chg_fcns.cmn'
	include 'ferret_cmn/xfr_grid.cmn'	! w/ mgrid_abstract
#else
	INCLUDE 'TMAP_FORMAT:TMAP_DIMS.PARM'
        INCLUDE 'TMAP_FORMAT:XTM_GRID.CMN_TEXT'
	INCLUDE 'FERRET_CMN:FERRET.PARM'
	INCLUDE 'FERRET_CMN:XGRID_CHG_FCNS.CMN'
	INCLUDE 'FERRET_CMN:XFR_GRID.CMN'	! W/ MGRID_ABSTRACT
#endif

* initialize
        IF (ifcn .LT. 0 .OR. iarg .LT. 1) THEN

           STOP 'gcf_get_axis_srcs'

        ELSEIF (ifcn .LE. gfcn_num_internal) THEN

* check the internal grid changing functions first
* is this a valid argument number?
	   IF (iarg .GT. gfcn_num_reqd_args(ifcn) )
	2	STOP 'gcf_get_axis_srcs'

* store info in local arrays to blur distinction between internal and external
* ... determine pointer to argument information
	   iptr1 = gfcn_arg_ptr(ifcn)
	   DO 10 idim = 1, 4
	      axis_will_be(idim) = gfcn_axis_will_be(idim,ifcn)
	      axis_implied_from(idim) =
     .			gfcn_axis_implied_from(idim,iptr1+iarg-1)
 10	   CONTINUE

	ELSE

	   IF (iarg .GT. efcn_get_num_reqd_args(ifcn) )
	2	STOP 'gcf_get_axis_srcs'

	   call efcn_get_axis_will_be(ifcn, axis_will_be)
	   call efcn_get_axis_implied_from(ifcn, iarg, axis_implied_from)

	ENDIF

* return information about argument iarg
*    axis_source may be returned as
*	pgc_impose_axis
*	pgc_merge_axis
*    or	pgc_ignore_axis

	DO 100 idim = 1, 4

	   IF (axis_will_be(idim) .EQ. pgc_axis_is_abstract)	THEN
	      axis_source(idim) = pgc_impose_axis

	   ELSEIF (axis_will_be(idim) .EQ. pgc_axis_is_normal) THEN
	      axis_source(idim) = pgc_impose_axis

	   ELSEIF (axis_will_be(idim) .EQ. pgc_axis_supplied_herein) THEN
	      axis_source(idim) = pgc_impose_axis

	   ELSEIF (axis_will_be(idim) .EQ. pgc_axis_implied_by_args) THEN

	      IF (axis_implied_from(idim)) THEN
	         axis_source(idim) = pgc_merge_axis
	      ELSE
	         axis_source(idim) = pgc_ignore_axis
	      ENDIF

	   ELSE
	      STOP 'Invalid axis merging information in GC func'

	   ENDIF

 100	CONTINUE

	RETURN
	END
