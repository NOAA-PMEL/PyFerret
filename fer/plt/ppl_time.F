	SUBROUTINE PPL_TIME( lo_val, hi_val )

* send the PLOT+ TIME command with WHOI-formatted dates to PLOT+

* programmer - steve hankin
* NOAA/PMEL, Seattle, WA - Tropical Modeling and Analysis Program

* V300:  10/93 - *sh* extracted logic from AXIS_FORMAT
* V420: 11/27/95 - 4-digit PLOT+ YEARS!!

#ifdef unix
	include 'ferret_cmn/ferret.parm'
	include 'ferret_cmn/plot_setup.parm'
	include 'ferret_cmn/xplot_setup.cmn'
* special equivalence needed only with DEC risc compiler: bug fix
        include 'ferret_cmn/xrisc.cmn'     ! for f77 bug
        CHARACTER rbuff(4)*14
        EQUIVALENCE ( rbuff, risc_buff )
#else
	INCLUDE 'FERRET_CMN:FERRET.PARM'
	INCLUDE 'FERRET_CMN:PLOT_SETUP.PARM'
        INCLUDE 'PPLV11INC:AXIS.INC/LIST'       ! with axis lengths	
#endif

* calling argument declarations:
	REAL*8	  lo_val, hi_val

* internal variable declarations:
	REAL*8	  tstart
	CHARACTER ABS_WHOI_DATE*14

* internal parameter declarations
	REAL*8		yrofsecs
	PARAMETER     ( yrofsecs = 60.*60.*24.*365. )

* set tstart 1 year before the beginning
	tstart = MAX( 0.0D0, lo_val - yrofsecs )

* TIME command (time axis limits)
#ifdef unix
              rbuff(1) = ABS_WHOI_DATE( lo_val )      ! f77 bug
              rbuff(2) = ABS_WHOI_DATE( hi_val )
              rbuff(3) = ABS_WHOI_DATE( tstart )
              WRITE ( ppl_buff, 3020 )  'W'//rbuff(1),
     .                                  'W'//rbuff(2),
     .                                  'W'//rbuff(3)
#else
	      WRITE ( ppl_buff, 3020 )	'W'//ABS_WHOI_DATE( lo_val ),
     .					'W'//ABS_WHOI_DATE( hi_val ),
     .					'W'//ABS_WHOI_DATE( tstart )
#endif
 3020	      FORMAT ('TIME ',A15,',',A15,',',A15)
	      CALL PPLCMD ( from, line, 0, ppl_buff, 1, 1 )
	RETURN
      END
