	INTEGER FUNCTION FIND_DSET_NUMBER( dname )

*
*
*  This software was developed by the Thermal Modeling and Analysis
*  Project(TMAP) of the National Oceanographic and Atmospheric
*  Administration's (NOAA) Pacific Marine Environmental Lab(PMEL),
*  hereafter referred to as NOAA/PMEL/TMAP.
*
*  Access and use of this software shall impose the following
*  obligations and understandings on the user. The user is granted the
*  right, without any fee or cost, to use, copy, modify, alter, enhance
*  and distribute this software, and any derivative works thereof, and
*  its supporting documentation for any purpose whatsoever, provided
*  that this entire notice appears in all copies of the software,
*  derivative works and supporting documentation.  Further, the user
*  agrees to credit NOAA/PMEL/TMAP in any publications that result from
*  the use of this software or in any product that includes this
*  software. The names TMAP, NOAA and/or PMEL, however, may not be used
*  in any advertising or publicity to endorse or promote any products
*  or commercial entity unless specific written permission is obtained
*  from NOAA/PMEL/TMAP. The user also understands that NOAA/PMEL/TMAP
*  is not obligated to provide the user with any support, consulting,
*  training or assistance of any kind with regard to the use, operation
*  and performance of this software nor to provide the user with any
*  updates, revisions, new versions or "bug fixes".
*
*  THIS SOFTWARE IS PROVIDED BY NOAA/PMEL/TMAP "AS IS" AND ANY EXPRESS
*  OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED
*  WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
*  ARE DISCLAIMED. IN NO EVENT SHALL NOAA/PMEL/TMAP BE LIABLE FOR ANY SPECIAL,
*  INDIRECT OR CONSEQUENTIAL DAMAGES OR ANY DAMAGES WHATSOEVER
*  RESULTING FROM LOSS OF USE, DATA OR PROFITS, WHETHER IN AN ACTION OF
*  CONTRACT, NEGLIGENCE OR OTHER TORTUOUS ACTION, ARISING OUT OF OR IN
*  CONNECTION WITH THE ACCESS, USE OR PERFORMANCE OF THIS SOFTWARE. 
*
*
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
* v500 *kob* 3/99 - modifications to allow datasets to start w/ a digit.
* v540 *acm* 10/01 increase length of ds_name
* v540 *acm* 10/01  remove VMS include statements
* v551 *acm* 1/03  Also check for extension .NC along with .DES, .CDF
* v580 *acm*11/04  Fix bug 1102; under linux, var[d=5coads] is read as dset=5

	include 'tmap_dims.parm'
#	include "tmap_dset.parm"
	include 'xdset_info.cmn_text'
	external xdset_info_data
	include	'ferret.parm'
	include 'gfdl_vms.parm'	

* calling argument declarations:
	CHARACTER*(*)	dname

* internal variable declarations:
	LOGICAL		TM_DIGIT, TM_HAS_STRING,
     .		        cdf_name, des_name, full_name_used, nc_name
	INTEGER		STR_CASE_BLIND_COMPARE, TM_LENSTR1,
     .			vax_code, dset, full_len, short_len
	CHARACTER	buff*256
	CHARACTER*13	TM_STRING, buff13, vbuff
        REAL*8          val

	IF ( TM_DIGIT( dname ) ) THEN

	   buff = dname				! buffer for AIX
* *kob* 3/99 - change the err= to go to 50 rather than 500 - this is so a
*              failed read a number is a clue that perhaps it is part of a name
*              of a dataset (eg 5coads) rather than just an error
* *kob* 3/30/99 - minor modification in logic forced by F90.  
c *kob* 4/97
* *acm* 11/04 current linux compiler reads a filename starting with digits
*             as just the digits with no error (10a.nc is read as dset=10)
*             Check explicitly whether the dset read is the same as buff

#ifdef MANDATORY_FORMAT_WIDTHS
	   READ (buff,*, ERR=50 ) dset
#else
	   READ (buff, '(I)', ERR=50 ) dset
#endif
           val = dset
           buff13 = TM_STRING(val)
           CALL TM_PACK_STRING(buff13, vbuff, 1,13, short_len)
	   vax_code = STR_CASE_BLIND_COMPARE( vbuff, buff(1:13) )
	   IF (  vax_code .NE. vms_str_success ) GOTO 50

	   GOTO 200
	ENDIF

* 5/93 locate "true" end to the given filename (.des, .cdf, and .nc are invisible)
* note: filenames are always in CAPS since they may be embedded in the var defn
 50	   full_len = TM_LENSTR1( dname )
	   IF ( full_len .GT. 4 ) THEN
	      cdf_name = dname(full_len-3:full_len) .EQ. '.cdf'
     .        .OR. dname(full_len-3:full_len) .EQ. '.CDF'
	      des_name = dname(full_len-3:full_len) .EQ. '.des'
     .        .OR. dname(full_len-3:full_len) .EQ. '.DES'
	      nc_name = dname(full_len-2:full_len) .EQ. '.nc'
     .        .OR. dname(full_len-2:full_len) .EQ. '.NC'
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
	   IF ( nc_name ) THEN
	      short_len = full_len - 3
	      full_name_used = .FALSE.
              cdf_name = .TRUE.
           ENDIF

* compare with filenames stored by the TMAP library
	   DO 100 dset = 1, maxdsets

	      IF ( ds_name(dset) .EQ. char_init256) GOTO 100

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
