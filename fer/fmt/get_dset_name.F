	SUBROUTINE GET_DSET_NAME( dset_num, name, len_text )

* get the name of the given data set if that is meaningful
* else get a string describing the nature of the name 

* programmer - steve hankin
* NOAA/PMEL, Seattle, WA - Tropical Modeling and Analysis Program
* written for VAX computer under VMS operating system

* V200	  1/4/89 - from GET_1_DSET_NAME: 
*		 - eliminated the possibility of multiple data set variables
* Unix/RISC port 5/91 *sh*: use ds_des_name; right justify long data set paths
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
	INTEGER		dset_num, len_text
	CHARACTER*(*)	name

* internal variable declarations:
	INTEGER		TM_LENSTR1, len_name

* initialize
	len_name  = LEN( name )

* consider the options
	IF     ( dset_num .EQ. pdset_irrelevant	)	THEN	! no name
	   name(1:len_name) = 'N/A'
	   len_text = 3

	ELSE
	   len_text = TM_LENSTR1( ds_des_name( dset_num ) )
           IF ( len_text .GT. len_name ) THEN
* ... too long to fit
              name = ds_des_name(dset_num)(len_text-len_name+1:)
              len_text = len_name
           ELSE
              name = ds_des_name( dset_num )
           ENDIF

	ENDIF

	RETURN
	END
