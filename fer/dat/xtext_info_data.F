	BLOCK DATA XTEXT_INFO_DATA

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
* pre-defined text strings

* programmer - steve hankin
* NOAA/PMEL, Seattle, WA - Tropical Modeling and Analysis Program
* written for VAX computer under VMS operating system
*
* revision 0.0 - 1/30/87
* revision 0.1 - 3/18/87 - added fvar_name_code (removed from *VARIABLES)
* revision 0.2 - 7/13/87 - added axis titles from ASCII_LIMITS
* revision 0.3 - 2/24/88 - added AIRT and axis_orients
* revision 0.4 - 3/18/88 - added date_labels and date_str_len
* revision 0.5 - 4/4/88  - added unit_text
* revision 0.6 -11/22/88 - new TM IO library took over axis_orients
* V300:  2/3/93 - added verify_modes
* V301:    1/94 - changed MODE VERIFY "FILES" to MODE VERIFY "DEFAULT" 
* Linux Port 1/97 *kob* - Added a preprocessor include for tmap_dset.parm
*			  because it needed a preprocessor.

#ifdef unix
#include "tmap_dset.parm"
	include	'ferret.parm'
	include	'xtext_info.cmn'
#else
	INCLUDE	'TMAP_FORMAT:TMAP_DSET.PARM'
	INCLUDE	'FERRET_CMN:FERRET.PARM'
	INCLUDE	'FERRET_CMN:XTEXT_INFO.CMN'
#endif

	DATA	ww_dim_name( x_dim )	/ 'X' /,
     .		ww_dim_name( y_dim )	/ 'Y' /,
     .		ww_dim_name( z_dim )	/ 'Z' /,
     .		ww_dim_name( t_dim )	/ 'T' /

	DATA	ss_dim_name( x_dim )	/ 'I' /,
     .		ss_dim_name( y_dim )	/ 'J' /,
     .		ss_dim_name( z_dim )	/ 'K' /,
     .		ss_dim_name( t_dim )	/ 'L' /

* pre-defined file variables
	DATA	fvar_name_code	( ptemp ) / 'TEMP'	/,
     .		fvar_name_code	( psalt ) / 'SALT'	/,
     .		fvar_name_code	( pu    ) / 'U'		/,
     .		fvar_name_code	( pv    ) / 'V'		/,
     .		fvar_name_code	( pw    ) / 'W'		/,
     .		fvar_name_code	( ptaux ) / 'TAUX'	/,
     .		fvar_name_code	( ptauy ) / 'TAUY'	/,
     .		fvar_name_code	( ppsi  ) / 'PSI '	/,
     .		fvar_name_code	( pairt ) / 'AIRT'	/

* axis names used to document outputs
	DATA	axis_title(x_dim)	/'LONGITUDE'/,	
     .		axis_title_len(x_dim)	/9/,
     .		axis_title(y_dim)	/'LATITUDE' /,
     .		axis_title_len(y_dim)	/8/,
     .		axis_title(z_dim)	/'DEPTH'    /,
     .		axis_title_len(z_dim)	/5/,
     .		axis_title(t_dim)	/'TIME'     /,	
     .		axis_title_len(t_dim)	/4/

* label formatting information
	DATA	date_labels ( 1 )	/ 'years'   /,
     .		date_labels ( 2 )	/ 'months'  /,
     .		date_labels ( 3 )	/ 'days'    /,
     .		date_labels ( 4 )	/ 'hours'   /,
     .		date_labels ( 5 )	/ 'minutes' /,
     .		date_labels ( 6 )	/ 'seconds' /
	DATA	date_str_len( 1 )	/  4 /,		! yyyy
     .		date_str_len( 2 )	/  8 /,		! mmm-yyyy
     .		date_str_len( 3 )	/ 11 /,		! dd-mmm-yyyy
     .		date_str_len( 4 )	/ 14 /,		! dd-mmm-yyyy:hh
     .		date_str_len( 5 )	/ 17 /,		! dd-mmm-yyyy:hh:mm
     .		date_str_len( 6 )	/ 20 /		! dd-mmm-yyyy:hh:mm:ss

* computed variable units
	DATA	unit_text( punot_app	)/ ' ' /,		! N/A
     .		unit_text( pudiffusion	)/ 'cm**2/sec' /,
     .		unit_text( puheat_flux	)/ 'watt/m**2' /,
     .		unit_text( putemp_dt	)/ 'deg. C/month' /,
     .		unit_text( puvel_dt	)/ 'cm/sec/month' /,
     .		unit_text( pupressure	)/ 'dynes/cm**2' /,
     .		unit_text( pudensity	)/ 'gm/cm**3' /,
     .		unit_text( pupres_dx	)/ 'dynes/cm**3' /,
     .		unit_text( puppt	)/ 'ppt' /,
     .		unit_text( puvelocity	)/ 'cm/sec' /,
     .		unit_text( putemperature)/ 'deg. C' /

* arguments for mode verify
        DATA   verify_modes(1) / 'DEFAULT' /,
     .         verify_modes(2) / 'ALL' /,
     .         verify_modes(3) / 'ALWAYS' /

	END
