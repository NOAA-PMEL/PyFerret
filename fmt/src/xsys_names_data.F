	BLOCK DATA XSYS_NAMES_DATA

* initialize data for indicated COMMON block

* programmer - steve hankin
* NOAA/PMEL, Seattle, WA - Tropical Modeling and Analysis Program
* written for VAX computer under VMS operating system
*
* revision 0.00 -  4/30/91

#ifdef unix
	include 'tmap_format/xsys_names.cmn_text'
#else
	INCLUDE 'TMAP_FORMAT:XSYS_NAMES.CMN_TEXT'
#endif

      DATA	tmap_sets        / 'FER_DESCR' /,
     .          tmap_tsteps      / 'FER_DATA'  /,
     .          tmap_grids       / 'FER_GRIDS' /,
     .          tmap_model_runs  / 'FER_MODEL_RUNS' /

      END
