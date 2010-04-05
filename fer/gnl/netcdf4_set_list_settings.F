	SUBROUTINE NETCDF4_SET_LIST_SETTINGS (status)
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
*  updates, revisions, new versions or 'bug fixes'.
*
*  THIS SOFTWARE IS PROVIDED BY NOAA/PMEL/TMAP 'AS IS' AND ANY EXPRESS
*  OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED
*  WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
*  ARE DISCLAIMED. IN NO EVENT SHALL NOAA/PMEL/TMAP BE LIABLE FOR ANY SPECIAL,
*  INDIRECT OR CONSEQUENTIAL DAMAGES OR ANY DAMAGES WHATSOEVER
*  RESULTING FROM LOSS OF USE, DATA OR PROFITS, WHETHER IN AN ACTION OF
*  CONTRACT, NEGLIGENCE OR OTHER TORTUOUS ACTION, ARISING OUT OF OR IN
*  CONNECTION WITH THE ACCESS, USE OR PERFORMANCE OF THIS SOFTWARE. 
*
*
* Get the settings for netcdf4 output datasets and store them in common.

* programmer - Ansley Manke
* NOAA/PMEL, Seattle, WA - Thermal Modeling and Analysis Program
*
* revision 0.0 - 1/13/2010

* Read info from qualilifiers for LIST netCDF4: NCFORMAT,
* XCHUNK,YCHUNK,ZCHUNK,TCHUNK,DEFLATE,SHUFFLE,ENDIAN

	include	'ferret.parm'
	include	'slash.parm'
	include 'xcontext.cmn'
	include 'xprog_state.cmn'
	include 'errmsg.parm'
        include 'netcdf4_settings.cmn'

* calling argument declarations
	INTEGER		status

* internal variable declarations
	INTEGER  STR_SAME, loc, slen
	CHARACTER buff*128

* initialize netcdf4-type stuff. If there has been a 
* SET LIST/NCFORMAT/ENDIAN/DEFLATE/SHUFFLE then these contain those values,
* otherwise they are set to the defaults

	netcdf4_type = netcdf4_type_set
	xchunk_size  = xchunk_size_set
	ychunk_size  = ychunk_size_set
	zchunk_size  = zchunk_size_set
	tchunk_size  = tchunk_size_set
	deflate_lev  = deflate_lev_set
	shuffle_flag = shuffle_flag_set
	endian_code  = endian_code_set

* /NCFORMAT for netCDF4 types ?
* NETCDF4 and CLASSIC, or 4=NETCDF4, 3=CLASSIC, 64BIT_OFFSET(?)

        loc = qual_given( slash_set_list_ncformat )
	IF (loc .GT. 0) THEN 
	   CALL EQUAL_STRING(
     .                   cmnd_buff(qual_start(loc):qual_end(loc)),
     .                   buff, status )              
           IF ( status .NE. ferr_ok ) GOTO 5000
	   IF (STR_SAME(buff(1:3), 'CLA') .EQ. 0) THEN
              netcdf4_type = 3
	   ELSEIF (STR_SAME(buff(1:1), '3') .EQ. 0) THEN
              netcdf4_type = 3
	   ELSEIF (STR_SAME(buff(1:7), 'NETCDF4') .EQ. 0) THEN
              netcdf4_type = 4
	   ELSEIF (STR_SAME(buff(1:1), '4') .EQ. 0) THEN
              netcdf4_type = 4
	   ELSEIF (STR_SAME(buff(1:3), '64B' ) .EQ. 0) THEN ! NF_64BIT_OFFSET
              netcdf4_type = 6
	   ELSEIF (STR_SAME(buff(1:1), '6' ) .EQ. 0) THEN
              netcdf4_type = 6
           ELSE
	      GOTO 5100
	   ENDIF
	   netcdf4_type_set = netcdf4_type
	ENDIF

* /XCHUNK,YCHUNK,ZCHUNK,TCHUNK for netCDF4 chunking?

	loc = qual_given( slash_set_list_xchunk )
	IF (loc .GT. 0) THEN 
	   CALL EQUAL_STRING(
     .                   cmnd_buff(qual_start(loc):qual_end(loc)),
     .                   buff, status )              
           READ (buff, *) xchunk_size
	   IF (xchunk_size .LE. 0) GOTO 5200
	   xchunk_size_set = xchunk_size
	ENDIF
 
        loc = qual_given( slash_set_list_ychunk )
	IF (loc .GT. 0) THEN 
	   CALL EQUAL_STRING(
     .                   cmnd_buff(qual_start(loc):qual_end(loc)),
     .                   buff, status )
           IF ( status .NE. ferr_ok ) GOTO 5000
           READ (buff, *) ychunk_size
	   IF (ychunk_size .LE. 0) GOTO 5200
	   ychunk_size_set = ychunk_size
	ENDIF

        loc = qual_given( slash_set_list_zchunk )
	IF (loc .GT. 0) THEN 
	   CALL EQUAL_STRING(
     .                   cmnd_buff(qual_start(loc):qual_end(loc)),
     .                   buff, status )
           IF ( status .NE. ferr_ok ) GOTO 5000
           READ (buff, *) zchunk_size
	   IF (zchunk_size .LE. 0) GOTO 5200
	   zchunk_size_set = zchunk_size
	ENDIF
	
        loc = qual_given( slash_set_list_tchunk )
	IF (loc .GT. 0) THEN 
	   CALL EQUAL_STRING(
     .                   cmnd_buff(qual_start(loc):qual_end(loc)),
     .                   buff, status )
           IF ( status .NE. ferr_ok ) GOTO 5000
           READ (buff, *) tchunk_size
	   IF (tchunk_size .LE. 0) GOTO 5200
	   tchunk_size_set = tchunk_size
	ENDIF

* DEFLATE 
* Can set compression level though all indicatins are that value=1 is best
* deflate=0 means no deflate

        loc = qual_given( slash_set_list_DEFLATE )
	IF (loc .GT. 0) THEN 
	   CALL EQUAL_STRING(
     .                   cmnd_buff(qual_start(loc):qual_end(loc)),
     .                   buff, status )
           IF ( status .NE. ferr_ok ) GOTO 5000
	   IF (STR_SAME(buff, ' ' ) .EQ. 0) THEN 
	      deflate_lev = 1
	   ELSE
              READ (buff, *) deflate_lev
	   ENDIF
	   IF (deflate_lev .LT. 0 .OR. deflate_lev.GT.9) GOTO 5300
	   deflate_lev_set = deflate_lev
	ENDIF

* SHUFFLE 
* Shuffle = 0 or 1

	loc = qual_given( slash_set_list_shuffle )
	IF (loc .GT. 0) THEN 
	   CALL EQUAL_STRING(
     .                   cmnd_buff(qual_start(loc):qual_end(loc)),
     .                   buff, status )
           IF ( status .NE. ferr_ok ) GOTO 5000
	   IF (STR_SAME(buff, ' ' ) .EQ. 0) THEN 
	      shuffle_flag = 1
	   ELSE
              READ (buff, *) shuffle_flag
	   ENDIF
	   IF (shuffle_flag .NE. 0 .AND. shuffle_flag.NE.1) GOTO 5400
	   shuffle_flag_set = shuffle_flag
	ENDIF

! ENDIAN: native, little, big

        loc = qual_given( slash_set_list_endian )
	IF (loc .GT. 0) THEN 
	   CALL EQUAL_STRING(
     .                   cmnd_buff(qual_start(loc):qual_end(loc)),
     .                   buff, status )
           IF ( status .NE. ferr_ok ) GOTO 5000
	   IF (STR_SAME(buff, ' ' ) .EQ. 0) THEN 
	      endian_code = 0 
	   ELSEIF (STR_SAME(buff(1:3), 'LIT') .EQ. 0) THEN
              endian_code = 1 ! nf_endian_little
	   ELSEIF (STR_SAME(buff(1:3), 'BIG') .EQ. 0) THEN
              endian_code = 2 ! nf_endian_big
	   ELSEIF (STR_SAME(buff(1:3), 'NAT') .EQ. 0) THEN
              endian_code = 0 ! nf_endian_native
	   ELSE
              GOTO 5500
	   ENDIF
	   
	   endian_code_set = endian_code
	ENDIF

* No checks for illegal compression or chunking requested on netCDF3 or 
* 64BIT_OFFSET file types. Settings may be made in more than one SET LIST/ 
* commands.  Need those kinds of checks when we go to make use of them.

	status = ferr_ok

 1000	CONTINUE
	RETURN

* error messages

 5000	CALL ERRMSG( ferr_invalid_command, status,
     .  'Error reading parameters in netcdf4_list_settings', *1000 )
 5100	CALL ERRMSG( ferr_invalid_command, status,
     . '/NCFORMAT may take values NETCDF4, CLASSIC, 4, 3, 64BIT_OFFSET', 
     . *1000 )
 5200	CALL ERRMSG( ferr_invalid_command, status,
     .		'/*CHUNK= values must be > 0', *1000 )
 5300	CALL ERRMSG( ferr_invalid_command, status,
     .		'/DEFLATE level must be between 0 and 9', *1000 )
 5400	CALL ERRMSG( ferr_invalid_command, status,
     .		'/SHUFFLE must take the value 0 or 1', *1000 )
 5500	CALL ERRMSG( ferr_invalid_command, status,
     . '/ENDIAN may be NATIVE, LITTLE or BIG', *1000 )
	END
