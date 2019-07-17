	SUBROUTINE DSG_ID_CX(cx_in, nfeatures)

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
* When a plot is colored by a feature-variable we extend the value
* ifeature along the obs axis. The group-context used for plot labels
* thinks that the plot is along the E axis but it is along the obs direction 
* Set the cx_plot index and/or world limits.

* NOAA/PMEL, Seattle, WA - Tropical Modeling and Analysis Program
* V741 8/18 *acm* color-by-feature

	include 'tmap_dims.parm'
	include 'ferret.parm'
	include 'xcontext.cmn'

* calling argument declarations:
	INTEGER cx_in, nfeatures

* When color-by-id, the feature-direction context has not been set in cx_plot.
* get it from the command context, where the user may have set a limit on E or M

	IF ( cx_lo_ss(cx_cmnd, e_dim) .NE. unspecified_int4) THEN
 	   cx_lo_ss(cx_plot, e_dim) = cx_lo_ss(cx_cmnd, e_dim)
	   cx_hi_ss(cx_plot, e_dim) = cx_hi_ss(cx_cmnd, e_dim)
	   cx_lo_ww(e_dim, cx_plot) = cx_lo_ss(cx_cmnd, e_dim)
	   cx_hi_ww(e_dim, cx_plot) = cx_hi_ss(cx_cmnd, e_dim)
	ELSE
 	   
	   IF ( cx_lo_ww(e_dim, cx_cmnd) .NE. unspecified_val8) THEN
	      cx_lo_ww(e_dim, cx_plot) = cx_lo_ww(e_dim, cx_cmnd)
	      cx_hi_ww(e_dim, cx_plot) = cx_lo_ww(e_dim, cx_cmnd)
	      cx_lo_ss(cx_plot, e_dim) = cx_lo_ww(e_dim, cx_cmnd)
	      cx_hi_ss(cx_plot, e_dim) = cx_lo_ww(e_dim, cx_cmnd)
	   ELSE	   
	      cx_lo_ss(cx_plot, e_dim) = 1
	      cx_hi_ss(cx_plot, e_dim) = nfeatures
	      cx_lo_ww(e_dim, cx_plot) = 1
	      cx_hi_ww(e_dim, cx_plot) = nfeatures
	   ENDIF

	ENDIF

* When color-by-id, the time context has not been set in cx_plot.
* get it from the context passed in

	IF ( cx_lo_ss(cx_plot, t_dim) .EQ. unspecified_int4) THEN
	   IF ( cx_lo_ss(cx_in, t_dim) .NE. unspecified_int4) THEN
 	      cx_lo_ss(cx_plot, t_dim) = cx_lo_ss(cx_in, t_dim)
	      cx_hi_ss(cx_plot, t_dim) = cx_hi_ss(cx_in, t_dim)
	   ENDIF
	ENDIF
 	   
	IF ( cx_lo_ww(t_dim, cx_plot) .EQ. unspecified_val8 
     .     .OR. cx_lo_ww(t_dim, cx_plot) .EQ. unspec_val8_2) THEN
	   IF ( cx_lo_ww(t_dim, cx_in) .NE. unspecified_val8) THEN
	      cx_lo_ww(t_dim, cx_plot) = cx_lo_ww(t_dim, cx_in)
	      cx_hi_ww(t_dim, cx_plot) = cx_hi_ww(t_dim, cx_in)
	   ENDIF
	ENDIF

	RETURN
	END

