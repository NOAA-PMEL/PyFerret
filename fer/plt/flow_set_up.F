	SUBROUTINE FLOW_SET_UP (mr_list, cx_list, transpz)
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
* set up for flowline plots, determining if we need the cosine correction

* programmer - Ansley Manke -- from disp_set_up
* NOAA/PMEL, Seattle, WA - Tropical Modeling and Analysis Program
* 5/31/01

C replace includes with those from plot_set_up, 
C plus those starting w/ xbox_bounds:
#ifdef unix
	include 'tmap_dims.parm'
	include 'xtm_grid.cmn_text'
	external xgt_grid_data
	include 'ferret.parm'
	include 'xplot_setup.cmn'
	include 'xvariables.cmn'

	include 'vector_inc.decl'	! with need_cos
	include 'VECTOR.INC'		! with need_cos

*kob 1/97
	include 'xwindow_state.cmn'
#endif
* calling argument declarations:
	INTEGER		mr_list(*), cx_list(*)

* internal variable declarations:

	INTEGER	dim1, dim2, mvx, ctx, grid
        LOGICAL	GEOG_COS_FACTOR, trnspz, transpz

* initialize
	mvx  = mr_list( 1 )
	ctx  = cx_list  ( 1 )
	grid = mr_grid( mvx )

* get dimensions of data region 
	dim1 = plot_axis(1)
	dim2 = plot_axis(2)

* will we need COS(latitude) corrections ? ( the true dx may be dx*COS(lat) )
        IF ( dim1.EQ.x_dim .AND. dim2.EQ.y_dim ) THEN
           need_cos =     GEOG_COS_FACTOR(x_dim, grid)
     .              .AND. GEOG_COS_FACTOR(y_dim, grid)
        ELSE
           need_cos = .FALSE.
        ENDIF

        trnspz = transpz

	RETURN
	END
