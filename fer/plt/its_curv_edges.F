	LOGICAL FUNCTION ITS_CURV_EDGES(idim)

* are the curvilinear coordinates grid points or box edges for this dimension

* V450: 3/97

* calling argument declarations
	INTEGER idim

* internal variable declarations

#ifdef unix
	include 'ferret_cmn/xcurvilinear.cmn'
#else
	INCLUDE 'FERRET_CMN:XVARIABLES.CMN'
#endif
	
	IF ( idim .EQ. 1 ) THEN

	   ITS_CURV_EDGES = dim1_edges	! from COMMON

	ELSE

	   ITS_CURV_EDGES = dim2_edges	! from COMMON

	ENDIF
	RETURN
	END
