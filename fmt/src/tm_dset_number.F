	INTEGER FUNCTION TM_DSET_NUMBER( path_name )
*
* Find the data set number of the given name
* If the name does not match any initialized data sets return unspecified_int4
*
* Programmer Steve Hankin
* NOAA/PMEL, Seattle, WA - Tropical Modeling and Analysis Program
* written for VAX/VMS
*
* revision 0.00 - 10/23/89 - code lifted from TM_INIT_DSET by Mark Verschell

* arguments:
*	path_name	- full path name of descriptor file

* Argument definitions
	CHARACTER*(*)	path_name

* Parameter and common files
#ifdef unix
	include 'tmap_format/tmap_errors.parm'		! error codes
	include 'tmap_format/tmap_dims.parm'		! data set dimensions
	include 'tmap_format/gt_lib.parm'
	include 'tmap_format/xdset_info.cmn_text'		! data set common
	external xdset_info_data
#else
	INCLUDE 'TMAP_FORMAT:TMAP_ERRORS.PARM'		! Error codes
	INCLUDE 'TMAP_FORMAT:TMAP_DIMS.PARM'		! Data set dimensions
	INCLUDE 'TMAP_FORMAT:GT_LIB.PARM'
	INCLUDE 'TMAP_FORMAT:XDSET_INFO.CMN'		! Data set common
#endif

* Internal declarations
	INTEGER		set_cnt

* check to see if this descriptor has already been initialized
	DO 100 set_cnt = 1, maxdsets
	    IF (ds_des_name(set_cnt) .EQ. path_name) THEN
* ... found a match
	      TM_DSET_NUMBER = set_cnt
	      RETURN
	    ENDIF
  100	CONTINUE

* no match (but no serious error)
	TM_DSET_NUMBER = unspecified_int4
	RETURN
	END
