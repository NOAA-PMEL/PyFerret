	LOGICAL FUNCTION ITSA_FILL_TRANS( trans )

* is the indicated transformation a filler?

* V450: 5/97

* calling argument declarations
	INTEGER	trans

* internal variable declarations

#ifdef unix
	include 'ferret_cmn/ferret.parm'
#else
	INCLUDE 'FERRET_CMN:FERRET.PARM'
#endif


	ITSA_FILL_TRANS =    trans .EQ. trans_fill_ave
     .			.OR. trans .EQ. trans_fill_interp
     .			.OR. trans .EQ. trans_fill_near

	RETURN
	END
