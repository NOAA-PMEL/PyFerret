	SUBROUTINE AXIS_ENDS( plot_ax, dat_ax, grid, first, last, delta )

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
* set up PPLUS to know the correct axis endpoints and tic interval

* programmer - steve hankin
* NOAA/PMEL, Seattle, WA - Tropical Modeling and Analysis Program
* written for VAX computer under VMS operating system
*
* revision 0.0 - 7/21/87
* revision 0.1 - 2/29/88 - allowed for non-inverted vertical Z axis
* V300 10/93 - allow delta value to be passed into this routine
* Linux Port *kob* 12/96 - had to add "external RANGE" because RANGE is an
*			   intrinsic function for F90.
* V500 - *jd* 5.3.99 Add axis extrema symbols
*	 *sh* 6/99 - consolidated axis extrema symbols into AXIS_END_SYMS
*             7/99 - use 7 digits of output precision
*        *kob* 9/99 - fixed minor bug in which length of val_buff was one
*                     character too short.  this only showed up under Linux
#ifdef unix
	include 'ferret.parm'
	include 'plot_setup.parm'
#else
	INCLUDE 'FERRET_CMN:FERRET.PARM'
	INCLUDE 'FERRET_CMN:PLOT_SETUP.PARM'
#endif

* kob 12/96
	EXTERNAL RANGE
* calling argument declarations:
	LOGICAL		BKWD_AXIS
	CHARACTER*1	plot_ax		! X (horiz) or Y (vert) axis of plot
	INTEGER		dat_ax, grid	! X,Y,Z or T axis of data
	REAL*8		first, last, delta

* internal variable declarations:
	CHARACTER*45    val_buff
	REAL		x1, x2, uminr, umaxr, dint

* convert to single precision and reverse order if a depth plot on Y axis
	IF ( plot_ax .EQ. 'Y'  .AND.  BKWD_AXIS(dat_ax, grid) ) THEN
	   x2 = SNGL(first)
	   x1 = SNGL(last)
	ELSE
	   x1 = SNGL(first)
	   x2 = SNGL(last)
	ENDIF

* use Denbo's PPLUS routine to calculate the tic interval, dint
	IF ( delta .EQ. unspecified_val8 ) THEN
	   CALL RANGE( x1, x2, 10, uminr, umaxr, dint )
	ELSE
	   dint = delta
	ENDIF

* and send the actual limits requested together with this tic interval to PPLUS
	WRITE ( val_buff, '(3(E14.7,1X))' ) x1, x2, dint
	CALL PPLCMD ( from, line, 0, 
     .				plot_ax//'AXIS '//val_buff , 1, 1 )

* Define symbols identifying axis limits
	CALL AXIS_END_SYMS(plot_ax, first, last)

	RETURN
      END
