	SUBROUTINE GET_DSET_TITLE( dset, title, len )

* get the title of the given data set

* programmer - steve hankin
* NOAA/PMEL, Seattle, WA - Tropical Modeling and Analysis Program
* written for VAX computer under VMS operating system
*
* revision 0.0 - 12/19/89 - simplified from GET_1_DSET_TITLE
* Linux Port 1/97 *kob* - Added a preprocessor include for tmap_dset.parm
*			  because it needed a preprocessor.

#ifdef unix
	include 'tmap_format/tmap_dims.parm'
#	include "tmap_format/tmap_dset.parm"
	include 'tmap_format/xdset_info.cmn_text'
	external xdset_info_data
	include 'ferret_cmn/ferret.parm'
#else
	INCLUDE 'TMAP_FORMAT:TMAP_DIMS.PARM'
	INCLUDE 'TMAP_FORMAT:TMAP_DSET.PARM'
	INCLUDE 'TMAP_FORMAT:XDSET_INFO.CMN'
	INCLUDE 'FERRET_CMN:FERRET.PARM'
#endif

* calling argument declarations:
	INTEGER		dset, len, TM_LENSTR1
	CHARACTER*(*)	title

	   IF ( dset .EQ. pdset_irrelevant ) THEN
	      title = ' '
	      len = 1
	   ELSE
* ... title line from descriptor file
	      title = ds_title( dset )
	      len   = TM_LENSTR1( title )
	   ENDIF
	RETURN
	END
