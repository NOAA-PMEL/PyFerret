	BLOCK DATA XCDFERROR_DATA

* initialize data for indicated COMMON block

* programmer - steve hankin
* NOAA/PMEL, Seattle, WA - Tropical Modeling and Analysis Program

* revision 0.00 -  3/92
* *sh* 3/92
* bug fix 14apr94 *kob* : DATA cdferrmsg(NCEGLOB) had an assigned string 
*			 which was longer than the variable was declared.
* IBM port 14apr94 *kob*: External statments in netcdf.inc caused IBM compiler
*		        : xlf to blow up.  Same statments in also cause OSF f77
*			: compiler to blow. So...include instead of netcdf.inc
*			: the hacked file netcdf.inc.no_externals
#ifdef unix
	include 'tmap_format/netcdf.decl'
	include 'tmap_format/netcdf_no_externals.inc'
	include 'tmap_format/xcdferror.cmn_text'
#else
	INCLUDE 'TMAP_FORMAT:NETCDF.DECL'
	INCLUDE 'TMAP_FORMAT:NETCDF.INC'
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

