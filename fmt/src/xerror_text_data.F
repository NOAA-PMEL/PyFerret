	BLOCK DATA XERROR_TEXT_DATA

* error message strings
* programmer - steve hankin
* NOAA/PMEL, Seattle, WA - Tropical Modeling and Analysis Program
* written for VAX computer under VMS operating system
*
* revision 0.00 - 11/20/85
* revision 0.10 - 05/07/87 - added error for Too may Grid files, and changed
*			     end of file to include "BAD DESCRIPTOR"
* revision 0.20 - 08/10/88 - Reworded error messages, and added some new ones
* revision 0.21 - 12/15/88 - Added time line definition error
* revision 0.22 - 01/06/89 - Removed merr_unkline, changed merr_lineundef,
*                            removed references to "COMMON" in error messages
* Unix/RISC port *sh* 3/15/91: changed text "not mounted" -> "not on line"
* 3/12/92 *sh* : added netCDF support - changed text of merr_badsubscr
*                                     - and merr_gridpredef
* 3/12/93 *sh* : allow unlimited columns in EZ data set

#ifdef unix
	include 'tmap_format/tmap_errors.parm'		! parameter definitions
	include 'tmap_format/xerror_text.cmn_text'	! common declaration
#else
	INCLUDE 'TMAP_FORMAT:TMAP_ERRORS.PARM'		! parameter definitions
	INCLUDE 'TMAP_FORMAT:XERROR_TEXT.CMN_TEXT'	! COMMON declaration
#endif

* special error messages
	DATA
     .	msg_special(merr_interrupt - pspecial)
     .		/'interrupted                                       '/

* regular error messages
	DATA
     .	msg_regular(merr_notsupport - pregular)
     .		/'Attempt to access unsupported feature             '/
     .	msg_regular(merr_filpos - pregular)
     .		/'record in incorrect position                      '/,
     .	msg_regular(merr_tstep - pregular)
     .		/'STEP record indicates incorrect time step         '/,
     .	msg_regular(merr_expnum - pregular)
     .		/'STEP record indicates incorrect run number        '/,
     .	msg_regular(merr_modnum - pregular)
     .		/'STEP record indicates incorrect mod number        '/,
     .	msg_regular(merr_notmounted - pregular)
     .		/'non-existent or not on line                       '/,
     .  msg_regular(merr_stepform - pregular)
     .		/'STEP record not found                             '/,
     .  msg_regular(merr_eof - pregular)
     .		/'End Of File encountered                           '/,
     .  msg_regular(merr_nmlerr - pregular)
     .		/'Error in namelist record                          '/,
     .  msg_regular(merr_unkvar - pregular)
     .		/'Unknown variable code                             '/,
     .  msg_regular(merr_unkgrid - pregular)
     .		/'Unknown grid specification                        '/,
     .  msg_regular(merr_stpmtch - pregular)
     .		/'1st timestep in datafile different from descriptor'/,
     .	msg_regular(merr_filim - pregular)
     .		/'file limit reached                                '/,
     .	msg_regular(merr_nostep - pregular)
     .		/'time step not found in data set                   '/,
     .	msg_regular(merr_dsetlim - pregular)
     .		/'limit on number of data sets has been reached     '/,
     .	msg_regular(merr_varlim - pregular)
     .		/'limit on number of variables has been reached     '/,
     .	msg_regular(merr_outofgrid - pregular)
     .		/'Requested data range is outside of data set limits'/,
     .	msg_regular(merr_badsubscr - pregular)
     .		/'Subscript limits dont make sense'/,
     .	msg_regular(merr_auxlim - pregular)
     .		/'limit on number of auxiliary sets has been reached'/,
     .	msg_regular(merr_grfillim - pregular)
     .		/'limit on number of grid files has been reached    '/,
     .	msg_regular(merr_gridlim - pregular)
     .		/'limit on number of grids has been reached         '/,
     .	msg_regular(merr_linelim - pregular)
     .		/'limit on number of grid lines has been reached    '/,
     .	msg_regular(merr_keynf - pregular)
     .		/'requested key not found in search list            '/,
     .	msg_regular(merr_gridpredef - pregular)
     .		/'attempt to redefine grid or attribute             '/
* regular error messages - continued
	DATA
     .	msg_regular(merr_lineundef - pregular)
     .		/'unknown line specification in grid file           '/,
     .	msg_regular(merr_linepredef - pregular)
     .		/'attempt to redefine line                          '/,
     .	msg_regular(merr_badgriddef - pregular)
     .		/'error in grid definition                          '/,
     .	msg_regular(merr_badlinedef - pregular)
     .		/'error in line definition                          '/,
     .	msg_regular(merr_badtimedef - pregular)
     .		/'error in time line definition                     '/,
     .	msg_regular(merr_linstorlim - pregular)
     .		/'limit on storage for coordinates has been reached '/,
     .	msg_regular(merr_nocoordlin - pregular)
     .		/'grid file defines line without giving coordinates '/,
     .	msg_regular(merr_unktype - pregular)
     .		/'unrecognized data type in FORMAT_RECORD           '/,
     .	msg_regular(merr_syntax - pregular)
     .		/'syntax error in string                            '/,
     .	msg_regular(merr_undefline - pregular)
     .		/'grid has unknown axis                             '/,
     .	msg_regular(merr_notimeinfo - pregular)
     .		/'Time axis info missing or incomplete for dataset  '/,
     .  msg_regular(merr_varform - pregular)
     .		/'VAR record not found                              '/,
     .  msg_regular(merr_varmtch - pregular)
     .		/'1st variable in datafile different from descriptor'/,
     .	msg_regular(merr_dsetnf - pregular)
     .		/'Data set requested is not initialized             '/,
     .	msg_regular(merr_stepnf-pregular)
     .		/'No stepfiles found for requested data set         '/,
     .	msg_regular(merr_badcolmn-pregular)
     .		/'# of columns not a multiple of # of vars          '/
	END
