	BLOCK DATA XGKS_DEVICES_DATA

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
* initialization for DEC GKS devices

* programmer - steve hankin
* NOAA/PMEL, Seattle, WA - Tropical Modeling and Analysis Program
* written for VAX computer under VMS operating system
*
* V200:  12/6/89

#ifdef unix
	include	'ferret_cmn/xgks_devices.cmn'
#else
	INCLUDE	'FERRET_CMN:XGKS_DEVICES.CMN'
#endif

	DATA	gks_dev_name(1) / 'DEFAULT'    	/,
     .		gks_dev_name(2) / 'VSII'    	/,
     .		gks_dev_name(3) / 'PS'    	/,
     .		gks_dev_name(4) / 'LN03P'    	/,
     .		gks_dev_name(5) / 'TEK4107'    	/,
     .		gks_dev_name(6) / 'TEK4014'    	/,
     .		gks_dev_name(7) / '&&&&&&&&'   	/,
     .		gks_dev_name(8) / '&&&&&&&&'   	/,
     .		gks_dev_name(9) / '&&&&&&&&'   	/,
     .		gks_dev_name(10)/ '&&&&&&&&'   	/

	DATA	gks_dev_wstype(1) / -999 /,
     .		gks_dev_wstype(2) / -999 /,
     .		gks_dev_wstype(3) / -999 /,
     .		gks_dev_wstype(4) / -999 /,
     .		gks_dev_wstype(5) / -999 /,
     .		gks_dev_wstype(6) / -999 /,
     .		gks_dev_wstype(7) / -999 /,
     .		gks_dev_wstype(8) / -999 /,
     .		gks_dev_wstype(9) / -999 /,
     .		gks_dev_wstype(10)/ -999 /

	DATA	meta_dev_name(1) / 'VSII'    	/,
     .		meta_dev_name(2) / 'PS'    	/,
     .		meta_dev_name(3) / 'LN03P'    	/,
     .		meta_dev_name(4) / 'TEK4107'    	/,
     .		meta_dev_name(5) / 'TEK4014'    	/,
     .		meta_dev_name(6) / '&&&&&&&&'   	/,
     .		meta_dev_name(7) / '&&&&&&&&'   	/,
     .		meta_dev_name(8) / '&&&&&&&&'   	/,
     .		meta_dev_name(9) / '&&&&&&&&'   	/,
     .		meta_dev_name(10)/ '&&&&&&&&'   	/
	END
