	INTEGER FUNCTION FIND_DSET_NUMBER( dname )

* find the data set number of set "dname"
* dname may be either the name (without directory pathname or .DES extension)
* or the number
* check that the number is valid before exiting

* programmer - steve hankin
* NOAA/PMEL, Seattle, WA - Tropical Modeling and Analysis Program
* written for VAX computer under VMS operating system
*
* revision 0.0 - 7/25/88
* revision 0.1 - 2/16/89 - ds_open --> ds_name.EQ.char_init30 to test open-ness
* Unix/RISC port 4/25/91 *sh*: increased size of ds_name
* V300: 5/93 *sh* - if the given name includes a .des or .cdf extension ignore
*		it when matching against data sets of these types
*		(TM_CLEAN_FILENAME doesn't store these extensions)
* V420 (IBM/AIX port of V411): 10/5/95 - buffer "READ(var,*"
*Linux Port 1/97 *kob* - Added a preprocessor include for tmap_dset.parm
*			  because it needed a preprocessor.
*	    4/97 *kob* - added ifdef for MANDATORY_FORMAT_WIDTHS on internal
*			 read because linux/nag f90 needed a format width.
#ifdef unix
	include 'tmap_format/tmap_dims.parm'
#	include "tmap_format/tmap_dset.parm"
	include 'tmap_format/xdset_info.cmn_text'
	external xdset_info_data
	include	'ferret_cmn/ferret.parm'
	include 'ferret_cmn/gfdl_vms.parm'	
#else
	INCLUDE 'TMAP_FORMAT:TMAP_DIMS.PARM'
	INCLUDE 'TMAP_FORMAT:TMAP_DSET.PARM'
	INCLUDE 'TMAP_FORMAT:XDSET_INFO.CMN'
	INCLUDE	'FERRET_CMN:FERRET.PARM'
	INCLUDE 'FERRET_CMN:GFDL_VMS.PARM'	
#endif

* calling argument declarations:
	CHARACTER*(*)	dname

* internal variable declarations:
	LOGICAL		TM_DIGIT, TM_HAS_STRING,
     .		        cdf_name, des_name, full_name_used
	INTEGER		STR_CASE_BLIND_COMPARE, TM_LENSTR1,
     .			vax_code, dset, full_len, short_len
	CHARACTER	buff*24

	IF ( TM_DIGIT( dname ) ) THEN

	   buff = dname				! buffer for AIX
c *kob* 4/97
#ifdef MANDATORY_FORMAT_WIDTHS
	   READ (buff,*, ERR=500 ) dset
#else
	   READ (buff, '(I)', ERR=500 ) dset
#endif
	ELSE

* 5/93 locate "true" end to the given filename (.des and .cdf are invisible)
* note: filenames are always in CAPS since they may be embedded in the var defn
	   full_len = TM_LENSTR1( dname )
	   IF ( full_len .GT. 4 ) THEN
	      cdf_name = dname(full_len-3:full_len) .EQ. '.cdf'
     .        .OR. dname(full_len-3:full_len) .EQ. '.CDF'
	      des_name = dname(full_len-3:full_len) .EQ. '.des'
     .        .OR. dname(full_len-3:full_len) .EQ. '.DES'
	   ELSE
	      cdf_name = .FALSE.
	      des_name = .FALSE.
	   ENDIF
	   IF ( cdf_name .OR. des_name ) THEN
	      short_len = full_len - 4
	      full_name_used = .FALSE.
	   ELSE
	      short_len = full_len
              full_name_used = .TRUE.
	   ENDIF

* compare with filenames stored by the TMAP library
	   DO 100 dset = 1, maxdsets

	      IF ( ds_name(dset) .EQ. char_init40 ) GOTO 100

* special name match checks to account for implied .cdf or .des   5/93
	      IF ( TM_HAS_STRING(ds_type(dset),'CDF')) THEN
	         vax_code = STR_CASE_BLIND_COMPARE( ds_name(dset),
     .						    dname(1:short_len) )
	         IF (  vax_code .EQ. vms_str_success
     .	        .AND. (cdf_name .OR. full_name_used) ) GOTO 200

	      ELSEIF ( TM_HAS_STRING(ds_type(dset),'GT')) THEN	
	         vax_code = STR_CASE_BLIND_COMPARE( ds_name(dset),
     .						    dname(1:short_len) )
	         IF (  vax_code .EQ. vms_str_success
     .	        .AND. (des_name .OR. full_name_used) ) GOTO 200

	      ELSE     ! EZ data set - any filename extension allowed
	         vax_code = STR_CASE_BLIND_COMPARE( ds_name(dset),
     .						    dname(1:full_len) )
	         IF (  vax_code .EQ. vms_str_success ) GOTO 200

	      ENDIF

 100	   CONTINUE
	   GOTO 500

	ENDIF

* check for valid number given
 200	IF ( dset .LT. 1 .OR. dset .GT. maxdsets ) GOTO 500
	IF ( ds_name(dset) .EQ. char_init40 ) GOTO 500

* success
	FIND_DSET_NUMBER = dset
	RETURN

* no match
 500	FIND_DSET_NUMBER = unspecified_int4
	RETURN

	END
