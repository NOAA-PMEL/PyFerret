	BLOCK DATA XEZ_INFO_DATA

* initialize data for indicated COMMON block
* programmer - mark verschell
* NOAA/PMEL, Seattle, WA - Tropical Modeling and Analysis Program
* written for VAX computer under VMS operating system
*
* revision 0.00 - 08/14/89

* COMMON/XEZ_INFO.....info for EZ_ data sets
#ifdef unix
	include 'tmap_format/tmap_dims.parm'
	include 'tmap_format/xez_info.cmn_text'
#else
	INCLUDE 'TMAP_FORMAT:TMAP_DIMS.PARM'
	INCLUDE 'TMAP_FORMAT:XEZ_INFO.CMN_TEXT'
#endif

	DATA	ez_mods		/maxdsets*.TRUE./

	END
