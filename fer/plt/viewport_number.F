	INTEGER FUNCTION VIEWPORT_NUMBER( view_name )

* identify the named viewport

* programmer - steve hankin
* NOAA/PMEL, Seattle, WA - Tropical Modeling and Analysis Program
* written for VAX computer under VMS operating system
*

* calling argument declarations:
	CHARACTER*(*)	view_name

* internal variable declarations:
	LOGICAL	MATCH_NAME
	INTEGER	ivp

#ifdef unix
	include	'ferret_cmn/ferret.parm'
	include	'ferret_cmn/gfdl_vms.parm'
	include 'ferret_cmn/xplot_state.cmn'
#else
	INCLUDE	'FERRET_CMN:FERRET.PARM'
	INCLUDE	'FERRET_CMN:GFDL_VMS.PARM'
	INCLUDE 'FERRET_CMN:XPLOT_STATE.CMN'
#endif

* revision 0.0 - 8/31/88
* V200: 10/29/89 - viewport 0 not find-able
*	  5/8/90 - bug: MATCH4 --> MATCH_NAME ("OV" doesn't match "OVER")

* search the viewport_names
	DO 10 ivp = 1, max_viewport
	   IF ( MATCH_NAME(view_name, vp_name(ivp)) ) THEN
	      VIEWPORT_NUMBER = ivp
	      RETURN
	   ENDIF

 10	CONTINUE

* given VIEWPORT name doesn't match possible options
	VIEWPORT_NUMBER = unspecified_int4
	RETURN

	END
