	SUBROUTINE NON_ARRAY_SUBSC( mr_list, nmr )

* transfer the lo:hi subscript information for the given list of memory-resident
* variables to non-array storage where it can be used for adjustable array
* dimensioning

* programmer - steve hankin
* NOAA/PMEL, Seattle, WA - Tropical Modeling and Analysis Program
* written for VAX computer under VMS operating system
*
* V200:  5/20/89

#ifdef unix
	include 'ferret_cmn/ferret.parm'
	include 'ferret_cmn/xvariables.cmn'
	include 'ferret_cmn/xmem_subsc.cmn'
#else
	INCLUDE 'FERRET_CMN:FERRET.PARM'
	INCLUDE 'FERRET_CMN:XVARIABLES.CMN'
	INCLUDE 'FERRET_CMN:XMEM_SUBSC.CMN'
#endif

* calling argument declarations:
	INTEGER nmr, mr_list(nmr)

* internal variable declarations:
	INTEGER idim, mr

	DO 100 idim = 1, 4
	DO 100 mr   = 1, nmr
	   lo_ss( mr, idim ) = mr_lo_ss( mr_list(mr), idim )
	   hi_ss( mr, idim ) = mr_hi_ss( mr_list(mr), idim )
 100	CONTINUE

	RETURN
	END
