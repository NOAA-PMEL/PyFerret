	BLOCK DATA XGKS_DEVICES_DATA

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
