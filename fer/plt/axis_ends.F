	SUBROUTINE AXIS_ENDS( plot_ax, dat_ax, grid, first, last, delta )

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

#ifdef unix
	include 'ferret_cmn/ferret.parm'
	include 'ferret_cmn/plot_setup.parm'
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
	CHARACTER*44    val_buff
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
	WRITE ( val_buff, '(3(E13.6,1X))' ) x1, x2, dint
	CALL PPLCMD ( from, line, 0, 
     .				plot_ax//'AXIS '//val_buff , 1, 1 )

	RETURN
      END
