	BLOCK DATA XREVISION_DATA

* pre-defined program revision and name

* programmer - steve hankin
* NOAA/PMEL, Seattle, WA - Tropical Modeling and Analysis Program
* written for VAX computer under VMS operating system
*
* V314 - 9/14/94 (extracted from ferret.F)
* V400 - 3/16/95 (at the joining with the GUI)
* V402 - 7/95 symbols (4.01), VAR[G=var2[d=dset]], etc. ...
*	 8/95 - The initialization of progname_mod moved to
*		xrevision_type_data.F for auto-configuring
* V410 - 8/95 ... just a name change so that the anonymous ftp looks good
* V411 - 9/95 repaired memory leaks: XGKS (clear_vp.F) and HDF (whdf.c)
* V420 - major changes
* V430 - added IF-THEN tests and wildcards
* V440 - 8/29/96 linked with GUI
* V441 - 10/15/96 identical to 4.40 but Sun and OSF V4.40's went out in debug
*		mode. This is to differentiate from them.
* V442 - 12/06/96 *kob* - gui bug fixes and quality control checks
* V445 - 1/14/97 *sh* - more gui bug fixes and uniform version numbering
* V500 - 4/5/97 *sh* - still in development at this date

#ifdef unix
	include	'ferret_cmn/xrevision.cmn'
#else
	INCLUDE	'FERRET_CMN:XREVISION.CMN'
#endif


	DATA program_name	/ 'FERRET' /
	DATA len_program_name	/  6       /
	DATA revision_level	/  5.00    /
!	DATA progname_mod	/ '(GUI BETA v1)' / ! ==> xrevision_type_data.F
!	DATA len_progname_mod	/  13       /	    ! ==> xrevision_type_data.F
	END
