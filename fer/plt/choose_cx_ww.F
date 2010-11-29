	SUBROUTINE CHOOSE_CX_WW( uvar, exp_cx, imp_cx, idim, ww_lo, ww_hi )

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
* given an explicit context which may or may not contain data and an
* implicit context which is always fleshed out determine the proper
* world coordinate limits for the specified axis

* programmer - steve hankin
* NOAA/PMEL, Seattle, WA - Tropical Modeling and Analysis Program
* written for VAX computer under VMS operating system
*
* V200:  5/19/89-8/6/89 (time representations)
* 	 3/2/90 - bug fix: variable, grid, not defined correctly
* V320:  3/2/95 - trim the 1/2 grid box fringe around CONTOUR and FILL plots
*               This is OK to do because /xlimits and /ylimits offer explicit
*               over-ride options
*       8/97 - *kob*  - check cx_calender rather than ww_low to see if time entered is a
*                       date string or a time step

#ifdef unix
	include 'ferret.parm'
	include 'command.parm'
	include 'xcontext.cmn'
	include 'xvariables.cmn'
	include 'xprog_state.cmn'
#else
	INCLUDE 'FERRET_CMN:FERRET.PARM'
	INCLUDE 'FERRET_CMN:COMMAND.PARM'
	INCLUDE 'FERRET_CMN:XCONTEXT.CMN'
	INCLUDE 'FERRET_CMN:XVARIABLES.CMN'
	INCLUDE 'FERRET_CMN:XPROG_STATE.CMN'
#endif

* calling argument declarations:
	INTEGER uvar, exp_cx, imp_cx, idim
	REAL*8	ww_lo, ww_hi

* internal variable declarations:
	INTEGER	imp_ndim, exp_ndim, imp_dim(4), exp_dim(4), grid,
     .		lo_edge, hi_edge
	REAL*8	TM_WORLD, SECS_TO_TSTEP, ww_low, ww_hiw

* initialize
* always use the grid carried with the data itself for ww/ss translations
	grid = cx_grid( imp_cx )

* default is to use the limits associated with the data itself
	   ww_lo = cx_lo_ww(idim,imp_cx)
	   ww_hi = cx_hi_ww(idim,imp_cx)
	   lo_edge =  box_lo_lim
	   hi_edge =  box_hi_lim

* 3/95 but for contour (and filled contour) plots we can only plot the edges
* up to the actual location of the point
        IF ( cmnd_num .EQ. cmnd_contour ) THEN
           ww_lo = MAX( ww_lo,
     .                  TM_WORLD(cx_lo_ss(imp_cx,idim),grid,idim,
     .                           box_middle)                    )
           ww_hi = MIN( ww_hi,
     .                  TM_WORLD(cx_hi_ss(imp_cx,idim),grid,idim,
     .                           box_middle)                    )
        ENDIF

* contour expression has its own limits imbedded ?
	IF ( uvar_given(idim,uvar) .LT. uvlim_needed ) RETURN

* contour expression depends on command limits - use 'em
* (test for unspecified in case limits came from file information)
	IF ( cx_by_ss(idim,exp_cx) ) THEN
	   IF ( cx_lo_ss(exp_cx,idim).EQ.unspecified_int4 ) THEN
	      RETURN
	   ELSE
	      ww_lo = MAX(TM_WORLD(cx_lo_ss(exp_cx,idim),grid,idim,
     .                    lo_edge),
     .			  cx_lo_ww(idim,imp_cx) )
	      ww_hi = MIN(TM_WORLD(cx_hi_ss(exp_cx,idim),grid,idim,
     .                    hi_edge),
     .			  cx_hi_ww(idim,imp_cx) )
	   ENDIF
	ELSE
	   IF ( cx_lo_ww(idim,exp_cx).EQ.unspecified_val8 ) THEN
	      RETURN
	   ELSE
* ... explicit limits can further restrict, but not expand
* 8/97 *kob* - check cx_calender for a date time rather than ww_low
	      ww_low = cx_lo_ww( idim, exp_cx )
	      ww_hiw = cx_hi_ww( idim, exp_cx )
	      IF ( idim.EQ.t_dim .AND. cx_calendar(exp_cx) ) THEN
	         ww_low = SECS_TO_TSTEP( grid, -ww_low )  
	         ww_hiw = SECS_TO_TSTEP( grid, -ww_hiw )  
	      ENDIF                                     
	      ww_lo = MAX( ww_low, ww_lo )
	      ww_hi = MIN( ww_hiw, ww_hi )
	   ENDIF
	ENDIF

	RETURN
	END
