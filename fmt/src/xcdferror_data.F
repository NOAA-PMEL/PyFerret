	BLOCK DATA XCDFERROR_DATA

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
* initialize data for indicated COMMON block

* programmer - steve hankin
* NOAA/PMEL, Seattle, WA - Tropical Modeling and Analysis Program

* revision 0.00 -  3/92
* *sh* 3/92
* bug fix 14apr94 *kob* : DATA cdferrmsg(NCEGLOB) had an assigned string 
*			 which was longer than the variable was declared.
* IBM port 14apr94 *kob*: External statments in netcdf.inc caused IBM compiler
*		        : xlf to blow up.  Same statments in also cause OSF f77
#include "netcdf.inc"
*			: the hacked file netcdf.inc.no_externals
#ifdef unix
	include 'netcdf.decl'
	include 'xcdferror.cmn_text'
#else
	INCLUDE 'TMAP_FORMAT:NETCDF.DECL'
#include "netcdf.inc"
	INCLUDE 'TMAP_FORMAT:XCDFERROR.CMN_TEXT'
#endif

      DATA cdferrmsg( NCEBADID ) 
     .               / 'Not a netcdf id'/
      DATA cdferrmsg( NCENFILE ) 
     .                / 'Too many netcdfs open'/
      DATA cdferrmsg( NCEEXIST ) 
     .                / 'netcdf file already exists'/
      DATA cdferrmsg( NCEINVAL ) 
     .                / 'Invalid Argument'/
      DATA cdferrmsg( NCEPERM ) 
     .                / 'Write to read only'/
      DATA cdferrmsg( NCENOTIN ) 
     .                / 'Operation not allowed in data mode'/
      DATA cdferrmsg( NCEINDEF ) 
     .                / 'Operation not allowed in define mode'/
      DATA cdferrmsg( NCECOORD ) 
     .                / 'Coordinates out of Domain'/
      DATA cdferrmsg( NCEMAXDS ) 
     .                / 'MAX netCDF dims exceeded'/
      DATA cdferrmsg( NCENAME ) 
     .                / 'String match to name in use'/
      DATA cdferrmsg( NCENOATT ) 
     .                / 'Attribute not found'/
      DATA cdferrmsg( NCEMAXAT ) 
     .                / 'MAX netCDF attributes exceeded'/
      DATA cdferrmsg( NCEBADTY ) 
     .                / 'Not a netcdf data type'/
      DATA cdferrmsg( NCEBADD ) 
     .                / 'Invalid dimension id'/
      DATA cdferrmsg( NCEUNLIM ) 
     .                / 'NCUNLIMITED in the wrong index'/
      DATA cdferrmsg( NCEMAXVS ) 
     .                / 'MAX netCDF variables exceeded'/
      DATA cdferrmsg( NCENOTVR ) 
     .                / 'Variable not found'/
      DATA cdferrmsg( NCEGLOB ) 
     .                / 'Action prohibited on global attrib.'/
      DATA cdferrmsg( NCENOTNC ) 
     .                / 'Not a netcdf file'/

      END

