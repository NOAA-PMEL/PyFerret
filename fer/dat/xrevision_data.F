	BLOCK DATA XREVISION_DATA

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
* V490 - 4/5/97 *sh* - still in development at this date
* v4.92 - 1/11/98 *kob* - upped version number from 4.91 to 4.92 for 
*                         source code release
* V500 - 12/98 *sh* - still in development as of this date
* V510 - 10/99 *sh*
* V511 - 05/00 *kob*
* V521 - *sh* minor bug fixes from V520
* V530 - 10/00 *sh* beta 1
* V534 - 9/01 *sh* alpha 1
* V550 - 7/02 *sh* alpha 1
* V551 - 2/03 *kob* beta1
* V580 - 12/04 *acm* 
* V581_01 8/12/05 *acm* minor revisions (not released)
* 5.812 10/5/05  *acm* bug fixes thru now...
* 5.813 12/13/05  *acm* fix for bug 1368
* 5.814  2/08/06  *acm* Fixes:
*                 further IF- parsing fixes, nested IF inside an IF clause that is not executed. 
*                 fix to restore scale defaults in tm_close_set.F (changes many bn outputs!)
*                 bug 1376, in labels and listings, if year is not listed because its modulo, 
*                 remove extra spaces
*                 bug 1077 had not been fixed for linux: handle arg-tests by QUERY as errors,
*                 if the test fails
*                 in NC reads, apply user-given offset to 1-D double precision variable, then 
*                 convert to single precision
* V6.00  4/25/06  Version 6.0
* V6.01  9/29/06  Version 6.01

	include	'xrevision.cmn'

	DATA program_name	/ 'FERRET' /
	DATA len_program_name	/  6       /
	DATA revision_level	/  6.01   /
!	DATA progname_mod	/ '(GUI BETA v1)' / ! ==> xrevision_type_data.F
!	DATA len_progname_mod	/  13       /	    ! ==> xrevision_type_data.F
	END
