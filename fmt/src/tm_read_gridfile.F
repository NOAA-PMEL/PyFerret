	SUBROUTINE TM_READ_GRIDFILE ( gfile_name, status)
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
* Reads in grid and line definitions and coordinates from a specified .GRD file
*
* revision 0.00 - 05/07/87 - by Mark Verschell for PMEL/TMAP
* revision 1.00 - 05/17/87 - by Mark Verschell for PMEL/TMAP
* revision 1.10 - 08/17/88 - added removal of grid/lines defined by bad file
* revision 1.11 - 08/18/88 - loops from max,1 to find empty storage
* revision 1.12 - 08/20/88 - old name GT_READ_GRIDFILE, called from TM_INIT_DSET
* revision 2.00 - 10/05/88 - new procedures for detection of duplicates
*			     Duplicated name/defn is ignored.
*			     Duplicated name, new defn is an error.
* revision 2.10 - 10/25/88 - Modulo flag added for lines definitions
* revision 2.11 - 11/21/88 - variable used checks changed to char_initXX
* revision 2.20 - 11/22/88 - Removed references to dset_num.
* revision 2.30 - 11/29/88 - True check for unknown and normal axis. Grid_line
*			     now flagged with mpsunkn or mpsnorm.
* revision 2.40 - 12/12/88 - Changes to make use of next_line_mem_pos, including
*			     resetting it if error occurs, and incrementing it
*			     only when line defined and checked for duplication.
* revision 2.41 - 12/13/88 - Added call TM_FIND_*_SLOT to determine open space
* revision 2.50 - 12/15/88 - Reads E/W as long. N/S as lat. for regular axis
* revision 2.60 - 12/19/88 - Reads time axis info, added TIME definition to the
*			     gridfile format to get t0 and tunit for timeaxis
* revision 2.70 - 01/13/89 - Any line that isn't a GRID/LINE/TIME/COORD/Comment 
*			     is not a warning message; doesn't read tunit from
*			     TIME line - gets it from un_convert
* revision 2.71 - 01/23/89 - Moved next_line_mem_pos save to before INQUIRE.
* revision 2.72 - 02/17/89 - Fixed next_line_mem_pos error (1 less than correct)
* revision 2.73 - 01/20/91 - added ifdef unix for default file
* Unix/RISC port - 4/24/91 *sh*: uses TM_INQ_PATH to locate grid file
*                              : changed "WARNING" to "Note"
*                              : allowed axis units of "none" or "n/a"
*                        10/91 : "grd" --> ".grd" for default extension
* bug fix - 10/2/91 - commented out statments 9000 and 9710 because there were
*                      no paths to these statements.  This was caught by ver.3
*                      of f77, but not ver. 2.1  ---kob
* bug fix - 6/24/92 - allow blank lines in gridfile
* bug fix - 1/4/93 *sh* - many error exits closed the unopened lunit
*                       - I added variable "opened"
* IBM Port- 4apr94 *kob* : readonly not allowed in file open statement
*			 : with IBM xlf compiler, added "ifdef" check
* *sh* 3/2/95 - informative messages issued through TM_NOTE (for GUI)
* *sh* 9/95 (for V420): axes need not have T0 to be valid.  Disable this check.
* 10/96 *kob* - Linux port - had to move variable assignments off of
*		declaration line.
* V500: 7/99 *sh* - bug fix: need consistent use of char_init16 on line_name
* V510: *sh* 8/99 bug fix: grid_out_prod not set for NORMAL axis
*			(see err500_del_var_corruption)
*	     4/00 - netCDF files now defining dynamic grids and axes. Need
*		to check among these for previously defined objects
* V541: *sh* 2/02 - minor change to set line_modulo_len
* V581 *acm* 6/05 - For fix to bug 1271, add flag line_shift_origin
* V65  *acm* 2/10- all warnings and notes to std error not std out

* arguments
	CHARACTER*(*)	gfile_name
	INTEGER*4	status

        include 'tmap_errors.parm'

	CALL TM_ERRMSG (merr_notsupport, status, 'TM_READ_GRIDFILE',
     .			no_descfile, no_stepfile,
     .			'Gridfiles no longer supported', no_errstring, *5000)

 5000	RETURN
	END

