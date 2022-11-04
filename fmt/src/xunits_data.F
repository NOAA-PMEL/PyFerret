	BLOCK DATA XUNITS_DATA

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
* unit names and conversion factors

* programmer - steve hankin
* NOAA/PMEL, Seattle, WA - Tropical Modeling and Analysis Program
* written for VAX computer under VMS operating system
*
* revision 0.00 - 04/15/88
* revision 0.01 - 11/21/88 - variable initialization changed to char_initXX
* revision 0.02 - 01/12/90 - added mb (kludge - not really convertible) *sh*
* revision 0.03 - 07/23/92 - added "layer" and "level" (unconvertible units)
*             (to eliminate error msg for these commonly used units)
* revision 0.04 - 12/19/94 - made mb non-convertible (1.0 instead of 0.01)
*			     and added unconvertible decibars
* 9/26/95 - interchanged the values (order) of pun_meters and pun_millibars
*		because TM_UNIT_ID was incorrectly interpreting "mbar"
* *kob* 10/96   - Linux port.  Linux didn't like tabs between "include"
*                 and what was being included.  removed them
* V5 *kob* 3/99 - add gregorian_ye[ar], year360, year366 units and also added
*              year365 as an alternate unit to the common year.
*    *sh* 7/99 - add alias "dbar" and change "month" definition to 1/12
*		 of 365.2425 day year 
*    *kob* 11/99 - add alternate unit "year" which is equal to a 
*                  gregorian_year
* *sh* 12/99 - add "degree" as an alternate name since exact match is now
*	required on names
* V510: 3/00 *sh* - added "minute" and "month"
* V520: 6/00 *sh* bug fix: exact matching in TM_UNIT_ID broke "seconds" match
*		         - add "hr" as an abbreviation for hours
*       1 day later: ops -- "hour" should be the standard, not "hr"
* V530: 2/01 *acm* add units for alternative calendars, secs per various
*		   years and 1/12 of those for "month"
* V532: 5/01 *sh* - added "metre"
* V554: 2/04 *acm*  default year length is 365.2425 days
* V570 *acm* 5/04  Add the calendar ALL_LEAP = 366_DAY, and units for the corresp. month.
* V71 12/16 *acm* Ticket 2158: working with true monthly time axes.
** 10/2022 *acm* Code cleanup: lengths of strings not to overflow length of character variable

* local variable declarations
!	INTEGER	i

	include 'tmap_dims.parm'
#include "tmap_dset.parm"
#include "gt_lib.parm"
	include 'xunits.cmn_text'


	DATA	un_name   ( 0)	/ ' '	/,	! FOR UNIDENTIFIED UNITS
     .		un_convert( 0)	/ 1.0 /

	DATA	un_name( 1)	/ 'cm'		/,
     .		un_name( 2)	/ 'km'		/,
     .		un_name(pun_meters)	/ 'm'		/,
     .		un_name(pun_degrees)	/ 'deg'		/,
     .		un_name( 5)	/ 'ft'		/,
     .		un_name( 6)	/ 'in'		/,
     .		un_name( 7)	/ 'mile'	/,
     .		un_name( 8)	/ 'mm'		/,
     .		un_name(pun_millibars)	/ 'mb'		/,
     .		un_name(pun_decibars)	/ 'dbar'	/,
     .		un_name(11)	/ char_init12	/,
     .		un_name(12)	/ char_init12	/,
     .		un_name(pun_instance)	/ "instance"	/,
     .		un_name(14)	/ 'level' 	/,
     .		un_name(15)	/ 'layer'	/

	DATA	un_name( -1)	/ 'sec'		/,
     .		un_name( -2)	/ 'min'		/,
     .		un_name( -3)	/ 'hour'	/,
     .		un_name( -4)	/ 'day'		/,
     .		un_name( -5)	/ 'mon'		/,
     .		un_name( -6)	/ 'yr'		/,
     .		un_name( -7)	/ 'M2 cycles'	/,
     .		un_name( -8)	/ 'year366'     /,
     .		un_name( -9)	/ 'year360'     /,
     .		un_name(-10)	/ 'gregorian_ye'/
     .		un_name(-11)	/ 'noleap_ye'/
     .		un_name(-12)	/ 'julian_ye'/
     .		un_name(-13)	/ 'mon_noleap'/
     .		un_name(-14)	/ 'mon_360'/
     .		un_name(-15)	/ 'mon_julian'/
     .          un_name(-16)    / 'mon_alleap'/

     .		un_name(-17)	/ 'tmon_gregori'/
     .		un_name(-18)	/ 'tmon_noleap'/
     .		un_name(-19)	/ 'tmon_360'/
     .		un_name(-20)	/ 'tmon_julian'/
     .		un_name(-21)	/ 'mon_alleap'/

	DATA	un_convert( 1)	/ .01		/,
     .		un_convert( 2)	/ 1000.		/,
     .		un_convert(pun_meters)	/ 1.0		/,
     .		un_convert( 4)	/ deg2m		/,
     .		un_convert( 5)	/ 0.3048	/,
     .		un_convert( 6)	/ 0.0254	/,
     .		un_convert( 7)	/ 1609.		/,
     .		un_convert( 8)	/ .001		/,
     .		un_convert(pun_millibars)	/ 1.0 /,! not convertible 12/94
     .		un_convert(pun_decibars)	/ 1.0 /,! kludge: not cnvertble
     .		un_convert(11)	/ real4_init	/,
     .		un_convert(12)	/ real4_init	/,
     .		un_convert(13)	/ real4_init	/,
     .		un_convert(14)	/ 1.0    	/, ! kludge for not convertible
     .		un_convert(15)	/ 1.0    	/  ! kludge for not convertible

	DATA	un_convert( -1)	/ 1.0		/,
     .		un_convert( -2)	/ 60.		/,
     .		un_convert( -3)	/ 3600.		/, ! sec per hour
     .		un_convert( -4)	/ 86400.	/, ! sec per day
     .		un_convert( -5)	/ 2629746.	/, ! 1/12 of 365.2425 year

c     .		un_convert( -6)	/ 31536000.	/, ! sec per 365 day year (OLD default)
     .		un_convert( -6)	/ 31556952.	/  ! sec per 365.2425 year

     .		un_convert( -7)	/ 44712.	/, ! sec per M2 cycle
     .		un_convert( -8)	/ 31622400.	/, ! sec per 366 day year
     .		un_convert( -9)	/ 31104000.	/, ! sec per 360 day year
     .		un_convert(-10)	/ 31556952.	/  ! sec per 365.2425 year

     .		un_convert(-11)	/ 31536000.	/, ! sec per 365.0 day year
     .		un_convert(-12)	/ 31557600.	/, ! sec per 365.25 day year

     .		un_convert(-13)	/ 2628000.	/, ! 1/12 of sec per 365 day year
     .		un_convert(-14)	/ 2592000.	/, ! 1/12 of sec per 360 day year
     .		un_convert(-15)	/ 2629800.	/, ! 1/12 of sec per 365.25 day year
     .		un_convert(-16)	/ 2635200.	/, ! 1/12 of sec per 366 day year

     .		un_convert(-17)	/ 2629746.	/, ! 1/12 of sec per 365.2425 year
     .		un_convert(-18)	/ 2628000.	/, ! 1/12 of sec per 365 day year
     .		un_convert(-19)	/ 2592000.	/, ! 1/12 of sec per 360 day year
     .		un_convert(-20)	/ 2629800.	/, ! 1/12 of sec per 365.25 day year
     .		un_convert(-21)	/ 2635200.	/  ! 1/12 of sec per 366 day year


	DATA	un_alt_name ( 1) / 'centim'	/,
     .		un_alt_subsc( 1) / 1 /,
     .
     .		un_alt_name ( 2) / 'kilom'	/,
     .		un_alt_subsc( 2) / 2 /,
     .
     .		un_alt_name ( 3) / 'milim'	/,
     .		un_alt_subsc( 3) / 8 /,
     .
     .		un_alt_name ( 4) / 'lat'	/,
     .		un_alt_subsc( 4) / pun_degrees /,
     .
     .		un_alt_name ( 5) / 'lon'	/,
     .		un_alt_subsc( 5) / pun_degrees /,
     .
     .		un_alt_name ( 6) / 'feet'	/,
     .		un_alt_subsc( 6) / 5 /,
     .
     .		un_alt_name ( 7) / 'foot'	/,
     .		un_alt_subsc( 7) / 5 /,
     .
     .		un_alt_name ( 8) / 'meter'	/,! after all other "xxxxmeters"
     .		un_alt_subsc( 8) / pun_meters /,
     .
     .		un_alt_name ( 9) / 'year365'/,
     .		un_alt_subsc( 9) / -6 /,
     .
     .		un_alt_name (10) / 'millibar'	/,
     .		un_alt_subsc(10) / pun_millibars /,
     .
     .		un_alt_name (11) / 'decibar'	/,
     .		un_alt_subsc(11) / pun_decibars /,
     .
     .		un_alt_name (12) / 'common_year'/,
     .		un_alt_subsc(12) / -6 /,
     .
     .		un_alt_name (13) / 'mbar'	/,
     .		un_alt_subsc(13) / pun_millibars /,
     .
     .		un_alt_name (14) / 'year'	/,
     .		un_alt_subsc(14) / -10 /,
     .
     .		un_alt_name (15) / 'degree'	/,
     .		un_alt_subsc(15) / pun_degrees /,
     .
     .		un_alt_name (16) / 'trumonth'	/,! use days
     .		un_alt_subsc(16) / -5 /,
     .
     .		un_alt_name (17) / 'month'	/,
     .		un_alt_subsc(17) / -5 /,
     .
     .		un_alt_name (18) / 'minute'	/,
     .		un_alt_subsc(18) / -2 /,
     .
     .		un_alt_name (19) / 'second'	/,
     .		un_alt_subsc(19) / -1 /,
     .
     .		un_alt_name (20) / 'hr'		/,
     .		un_alt_subsc(20) / -3 /,
     .
     .		un_alt_name (21) / 'metre'	/, ! those Brits ...
     .		un_alt_subsc(21) /  pun_meters /
!     .		un_alt_name (21) / char_init12	/,
!     .		un_alt_subsc(21) / 0 /

	END
