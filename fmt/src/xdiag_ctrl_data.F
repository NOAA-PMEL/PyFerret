	BLOCK DATA XDIAG_CTRL_DATA

* initialize data for indicated COMMON block

* programmer - steve hankin
* NOAA/PMEL, Seattle, WA - Tropical Modeling and Analysis Program
* written for VAX computer under VMS operating system
*
* revision 0.00 -  3/15/91
* 2/92 *sh* added ds_dset_type to support netCDF

* COMMON/XDIAG_CTRL.....variables to control diagnostic output

#ifdef unix
	include 'tmap_format/xdiag_ctrl.cmn_text'
#else
	INCLUDE 'TMAP_FORMAT:XDIAG_CTRL.CMN_TEXT'
#endif

      DATA	tmap_diag_on    /.FALSE./
      DATA      ds_dset_type    /'    '/
      END
