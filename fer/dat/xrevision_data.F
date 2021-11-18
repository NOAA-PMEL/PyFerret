	BLOCK DATA XREVISION_DATA

*  This software was developed by the Thermal Modeling and Analysis
*  Project(TMAP) of the National Oceanographic and Atmospheric
*  Administration''s (NOAA) Pacific Marine Environmental Lab(PMEL),
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
* V441 - 10/15/96 identical to 4.40 but Sun and OSF V4.40 went out in debug
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
* V6.02  1/05/07  Version 6.02
* V6.03  5/25/07  Version 6.03
* V6.04  7/02/07  Version 6.04
* V6.05  7/27/07  Version 6.05
* V6.06  8/08/07  Version 6.06 for LAS
* V6.07  8/24/07  Version 6.07 for LAS
* V6.08 10/12/07  Version 6.08
* V6.10  3/25/08  Version 6.1
* V6.11  4/02/08  Version 6.11 internal release
* V6.12  5/20/08  Version 6.12
* V6.13  8/28/08  Version 6.13
* V6.13  8/28/08  Version 6.14 GFDL see bug 1586,1038
* V6.15 11/07/08  Version 6.15 GFDL bugs 1608,1609
* V6.16 11/20/08  Version 6.16 GFDL bugs 1421,1523,1547,1522,1577 plus gif transparency
* V6.17  1/12/09  Version 6.17 GFDL bug  1476, bugs 1588, 1333, 1614 
* V6.18  1/21/09  Version 6.18 fix bug with CURV_TO_RECT, allowing subsetting
* V6.19  2/03/09  Version 6.19 fix for choosing colors and thicknesses when > 6 colors have been defined
* V6.2   5/0/09   V6.2 release
* V6.3   9/18/09  V6.3 release
* V6.4   11/20/09  V6.4 release
* V6.61   4/27/10  
* V6.62   5/14/10  release v6.62 this week: LAS changes  
* V6.64            BROWSE, bug fixes. Released 9/2010  
* V6.65  10/01/10
* V6.74   4/2011  to GFDL: double precision, netCDF4.2, fix ticket 1391 
* V6.84  12/2012  Ribbon plots, bug fixes.
* V6.9    4/2014  
* V6.91   internal, not released. Fixes on v6.9
* V6.92   internal, new color /key options any of 
*           /key=vertical,horizontal,centerlabel
* V6.921   internal, @WGT and defined axis for EZ reads
* V6.922   final tweaks to /key=centerlab, internal tag for SOCAT testing
* v6.923   fix ticket 2187
* v6.924   fix ticket 2186 
* v6.925   fix ticket 2194
* v6.926   fix ticket 2203, 2200
* v6.927   fix ticket 2205, 2206
* v6.93    release
* v6.931   to gfdl
* v6.932   to John Maurer, bug 2223 and tag v_6_9_32_GFDL23Dec
* v6.933   to GFDL tag v_6_9_33_GFDL27Jan15
* v6.94    Ferret release to go with LAS 8.4
* v6.941   Version number for post-v6.94 development
* v6.942   Version number for post-v6.94 development 5/8/2015
* v6.943   To GFDL  tag v_6_9_43_GFDL_2Jun15 tickets thru 2296
* v6.95    release for LAS8.4
* v6.951   New development
* v6.96    little release with @BIN @NBIN regrid; high-res time plots
* v6.961   To GFDL for fix to #2334
* v6.963   To GFDL in a PyFerret update, install on our servers
* v6.97 (working version)
* v7.0     7/2016
* v7.02    8/2016 Fixes graticules on time axes # 2466
* v7.02   12/2016 Merged true-monthly axis branch to the trunk
* v7.1     2/2017 Official release
* v7.11    4/19/2017 Tag before GFDL visit 4/2017
* v7.12    5/11/2017 
* v7.13    6/ 6/2017  fixes to tickets 2538, 2539

* v7.11    4/19/2017 Tag before GFDL visit 4/2017
* v7.12    5/11/2017 
* v7.13    6/ 6/2017  fixes to tickets 2538, 2539
* v7.20    7/11/2017 7.20 release
* v7.21    9 /2017 7.21 release
* v7.22    9/14/2017 7.22 dnamic coordinate storage
* v7.3     11/30/2017 Release 
* v7.31    12/06/2017 next working version
* v7.32     2/27/2018 next working version
* v7.4      4/10/2018 Release
* v7.42     7/10/2018 to gfdl
* v7.43     9/14/2018 bug-fix release
* v7.44    12/07/2018 minor release; back to using NetCDF 4.4.1.1 instead of 4.6.x
* v7.5      4/22/2019 Release
* v7.6     11/27/2019 bump DSG version number to 7.6 to distinguish from updated pre-DSG version (7.52)
* V7.61     7/23/2020 fixes and additions to 7.6
* V7.62     8/18/2020 TrajectoryProfile and TimeseriesProfile handled, but only as if they are profile
* v7.63     9/23/2020 TrajectoryProfile and TimeseriesProfile handled for plots, listings, not SAVE
*                     PLOT/ALONG=xy for maps of traj-prof, ts-prof, and also traj and prof data       
*                     PLOT/ALONG=T for TimeseriesProfile data

        IMPLICIT NONE
	include	'xrevision.cmn'

	DATA program_name	/ 'PyFerret' /
	DATA len_program_name	/  8         /
	DATA revision_level	/  7.65   /

	END
