	BLOCK DATA XERROR_TEXT_DATA

*
*
*  This software was developed by the Thermal Modeling and Analysis
*  Project(TMAP) of the National Oceanographic and Atmospheric
*  Administration's (NOAA) Pacific Marine Environmental Lab(PMEL),
*  hereafter referred to as NOAA/PMEL/TMAP.
*
*  Access and use of this software shall impose the following
*  obligations and understandings on the user. The user is granted the
*  right, without any fee or cost, to use, copy, modify, alter, enhance
*  and distribute this software, and any derivative works thereof, and
*  its supporting documentation for any purpose whatsoever, provided
*  that this entire notice appears in all copies of the software,
*  derivative works and supporting documentation.  Further, the user
*  agrees to credit NOAA/PMEL/TMAP in any publications that result from
*  the use of this software or in any product that includes this
*  software. The names TMAP, NOAA and/or PMEL, however, may not be used
*  in any advertising or publicity to endorse or promote any products
*  or commercial entity unless specific written permission is obtained
*  from NOAA/PMEL/TMAP. The user also understands that NOAA/PMEL/TMAP
*  is not obligated to provide the user with any support, consulting,
*  training or assistance of any kind with regard to the use, operation
*  and performance of this software nor to provide the user with any
*  updates, revisions, new versions or "bug fixes".
*
*  THIS SOFTWARE IS PROVIDED BY NOAA/PMEL/TMAP "AS IS" AND ANY EXPRESS
*  OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED
*  WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
*  ARE DISCLAIMED. IN NO EVENT SHALL NOAA/PMEL/TMAP BE LIABLE FOR ANY SPECIAL,
*  INDIRECT OR CONSEQUENTIAL DAMAGES OR ANY DAMAGES WHATSOEVER
*  RESULTING FROM LOSS OF USE, DATA OR PROFITS, WHETHER IN AN ACTION OF
*  CONTRACT, NEGLIGENCE OR OTHER TORTUOUS ACTION, ARISING OUT OF OR IN
*  CONNECTION WITH THE ACCESS, USE OR PERFORMANCE OF THIS SOFTWARE. 
*
*
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
* 11/99 *sh* - change "grid lines" to "axes" in error message
* V600 6/05 *acm* new error merr_attalready; attribute already defined
* V604 6/07 *acm* New merr_remote_open for errors opening remote datasets
* V697 12/15*acm* New merr_axis_att reporting errors in axis attributes, 
*                 e.g. calendars (For ticket 2332)
* v741 6/18 *acm* New merr_dsg_grid and merr_invalid, reporting DSG grid errors
* v741 7/18 *acm* New merr_dsg_struc, reporting DSG attribute errors
* v767 6/22 *acm* See issue 112: New merr_dimtoolarge, reporting dimension too 
*                 large for 4-byte integer indexing

	include 'tmap_errors.parm'		! parameter definitions
	include 'xerror_text.cmn_text'	! common declaration

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
     .		/'limit on number of axes has been reached          '/,
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
     .	msg_regular(merr_stepnf - pregular)
     .		/'No stepfiles found for requested data set         '/,
     .	msg_regular(merr_badcolmn - pregular)
     .		/'# of columns not a multiple of # of vars          '/,
     .	msg_regular(merr_attalready - pregular)
     .		/'attribute already defined                         '/,
     .	msg_regular(merr_badfileatt - pregular)
     .		/'attribute reading error                           '/,
     .	msg_regular(merr_nc_open - pregular)
     .		/'error opening netCDF file                         '/,
     .	msg_regular(merr_remote_open - pregular)
     .		/'error opening remote data file                    '/,
     .	msg_regular(merr_chunk_spec - pregular)
     .		/'error specifying chunk sizes                      '/
     .	msg_regular(merr_time_string - pregular)
     .		/'error creating time string                        '/
     .	msg_regular(merr_axis_att - pregular)
     .		/'error in axis attributes                          '/
     .	msg_regular(merr_dsg_grid - pregular )
     .		/'Coordinates inconsistent with FeatureType         '/
     .	msg_regular(merr_dsg_struc - pregular )
     .		/'Corrupted CF-DSG structure or attributes          '/
     .	msg_regular(merr_invalid - pregular )
     .		/'Invalid command                        '/
     .	msg_regular(merr_dimtoolarge - pregular )
     .		/'Dimension too large for 4-byte integer indexing   '/
	END
