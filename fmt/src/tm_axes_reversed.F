	INTEGER FUNCTION TM_AXES_REVERSED( var, axes )

* return an array of length 4 indicating containing the axes that are reversed
* and the value of the function is the number that are reversed

* ... to hide TMAP COMMON from FERRET

* programmer - steve hankin
* NOAA/PMEL, Seattle, WA - Tropical Modeling and Analysis Program
* written for VAX computer under VMS operating system
*
* V320: 2/16/94

#ifdef unix
	include 'tmap_format/tmap_dims.parm'
#include "tmap_format/tmap_dset.parm"
	include 'tmap_format/xdset_info.cmn_text'
#else
	INCLUDE 'TMAP_FORMAT:TMAP_DIMS.PARM'
	INCLUDE 'TMAP_FORMAT:TMAP_DSET.PARM'
	INCLUDE 'TMAP_FORMAT:XDSET_INFO.CMN_TEXT'
#endif

* calling argument declarations:
* *kob* 8/95 - changed decl. of axes from LOGICAL to INTEGER
	INTEGER	axes(4)
	INTEGER	var

* internal variable declarations
	INTEGER	i, number

* initialze
	number = 0

* check all the axes
* note that for netCDF files the array ds_ordering contains a list of only
* those dimensions that are relevant to the variable.  For example, the
* variable SST, which contains axes X (1), Y (2), and T (3) would have
* ds_ordering(1-4,SST_var) = 1,2,4,0
	DO 100 i = 1, 4
	   IF ( ds_ordering(i,var) .LT. 0 ) THEN
	      number = number + 1
	      axes(number) = ABS(ds_ordering(i,var))
	   ENDIF
 100	CONTINUE	   

	TM_AXES_REVERSED = number

	RETURN
	END
