	SUBROUTINE FORGET_PAST_CONTEXT_MODS ( context )

* erase/clear away indicators of when last modifications were made to the given
* context

* programmer - steve hankin
* NOAA/PMEL, Seattle, WA - Tropical Modeling and Analysis Program
* written for VAX computer under VMS operating system
*
* revision 0.0 - ???
* revision 0.1 - 7/26/88 - documentation re-write

#ifdef unix
	include	'ferret_cmn/ferret.parm'
	include	'ferret_cmn/xcontext.cmn'
#else
	INCLUDE	'FERRET_CMN:FERRET.PARM'
	INCLUDE	'FERRET_CMN:XCONTEXT.CMN'
#endif

* calling argument declarations:
	INTEGER	context

* internal variable declarations:
	INTEGER	idim

	DO 100 idim = 1,4
 100	cx_given( idim, context ) = .FALSE.

	RETURN
	END
