	BLOCK DATA XPLOT_SETUP_DATA

* initialization of COMMON / XPLOT_SETUP /
* programmer - steve hankin
* NOAA/PMEL, Seattle, WA - Tropical Modeling and Analysis Program
* written for VAX computer under VMS operating system

* revision 0.1 - 9/12/88 - reduced number of defined colors to 10
* V200: 12/21/89

#ifdef unix
	include 'ferret_cmn/ferret.parm'
	include 'ferret_cmn/xplot_setup.cmn'
#else
	INCLUDE 'FERRET_CMN:FERRET.PARM'
	INCLUDE 'FERRET_CMN:XPLOT_SETUP.CMN'
#endif

	DATA
     .		saved_ppl_yax	/ .FALSE. /,
     .		key_xspace0	/ 3.0 /,	! nominal X space for keys
     .		key_yspace0	/ 0.25 /,	! nominal Y space for keys
     .		key_labht0	/ 0.12 /,	! nominal label height for keys
     .		key_xorigin	/ 1.5  /,	! leftmost x loc. of keys
     .		key_yorigin	/ -0.85 /,	! y loc. of topmost keys
     .		key_xwhite_space/ 0.3  /,	! x space between keys
     .		key_ywhite_space/ 0.02 /,	! y space between keys
     .		key_line_frac   / 0.3 /,	! how much key for line specimen
     .		ovkey_xspace	/ 2.1 /,
     .		ovkey_yspace	/ 0.18 /,
     .		ovkey_labht	/ 0.08 /,
     .		ovkey_xorigin	/-1.1 /,
     .		ovkey_yorigin	/-0.6 /,
     .		ul_labht0	/0.12 /,
     .		ul_yspace0	/0.2  /,
     .		ul_xorigin	/ 0.0 /,	! measured from Y axis
     .		ul_yorigin	/ 0.6 /		! measured from top x axis

	END
