	BLOCK DATA XUNITS_DATA

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
* local variable declarations
	INTEGER	i

#ifdef unix
	include 'tmap_format/tmap_dims.parm'
#include "tmap_format/tmap_dset.parm"
	include 'tmap_format/gt_lib.parm'
	include 'tmap_format/xunits.cmn_text'
#else
	INCLUDE	'TMAP_FORMAT:TMAP_DIMS.PARM'
	INCLUDE	'TMAP_FORMAT:TMAP_DSET.PARM'
	INCLUDE	'TMAP_FORMAT:GT_LIB.PARM'
	INCLUDE 'TMAP_FORMAT:XUNITS.CMN_TEXT'
#endif

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
     .		un_name(pun_decibars)	/ 'decibar'	/,
     .		un_name(11)	/ char_init12	/,
     .		un_name(12)	/ char_init12	/,
     .		un_name(13)	/ char_init12	/,
     .		un_name(14)	/ 'level' 	/,
     .		un_name(15)	/ 'layer'	/

	DATA	un_name( -1)	/ 'sec'		/,
     .		un_name( -2)	/ 'min'		/,
     .		un_name( -3)	/ 'hour'	/,
     .		un_name( -4)	/ 'day'		/,
     .		un_name( -5)	/ 'mon'		/,
     .		un_name( -6)	/ 'yr'		/,
     .		un_name( -7)	/ 'M2 cycles'	/,
     .		un_name( -8)	/ char_init12	/,
     .		un_name( -9)	/ char_init12	/,
     .		un_name(-10)	/ char_init12	/

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
     .		un_convert( -3)	/ 3600.		/,	! sec per hour
     .		un_convert( -4)	/ 86400.	/,	! sec per day
     .		un_convert( -5)	/ 2628000.	/,	! sec per 730 hr month
     .		un_convert( -6)	/ 31536000.	/,	! sec per 365 day year
     .		un_convert( -7)	/ 44712.	/,	! sec per M2 cycle
     .		un_convert( -8)	/ real4_init	/,
     .		un_convert( -9)	/ real4_init	/,
     .		un_convert(-10)	/ real4_init	/

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
     .		un_alt_name ( 9) / 'year'	/,
     .		un_alt_subsc( 9) / -6 /,
     .
     .		un_alt_name (10) / 'millibar'	/,
     .		un_alt_subsc(10) / pun_millibars /,
     .
     .		un_alt_name (11) / 'decibar'	/,
     .		un_alt_subsc(11) / pun_decibars /,
     .
     .		un_alt_name (12) / char_init12	/,
     .		un_alt_subsc(12) / 0 /,
     .
     .		un_alt_name (13) / char_init12	/,
     .		un_alt_subsc(13) / 0 /,
     .
     .		un_alt_name (14) / char_init12	/,
     .		un_alt_subsc(14) / 0 /,
     .
     .		un_alt_name (15) / char_init12	/,
     .		un_alt_subsc(15) / 0 /,
     .
     .		un_alt_name (16) / char_init12	/,
     .		un_alt_subsc(16) / 0 /,
     .
     .		un_alt_name (17) / char_init12	/,
     .		un_alt_subsc(17) / 0 /,
     .
     .		un_alt_name (18) / char_init12	/,
     .		un_alt_subsc(18) / 0 /,
     .
     .		un_alt_name (19) / char_init12	/,
     .		un_alt_subsc(19) / 0 /,
     .
     .		un_alt_name (20) / char_init12	/,
     .		un_alt_subsc(20) / 0 /

	END
