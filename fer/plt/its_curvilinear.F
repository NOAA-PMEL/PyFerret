	LOGICAL FUNCTION ITS_CURVILINEAR( )

* are we being asked to do a curvilinear coordinate plot?

* V450: 3/97

* calling argument declarations

* internal variable declarations

#ifdef unix
	include 'ferret_cmn/xcurvilinear.cmn'
#else
	INCLUDE 'FERRET_CMN:XVARIABLES.CMN'
#endif
	
	ITS_CURVILINEAR = curvilinear	! from COMMON

	RETURN
	END
