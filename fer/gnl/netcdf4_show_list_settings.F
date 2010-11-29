	SUBROUTINE NETCDF4_SHOW_LIST_SETTINGS
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
* Get the settings for netcdf4 output datasets and show them as other
* SET LIST settings are shown

* programmer - Ansley Manke
* NOAA/PMEL, Seattle, WA - Thermal Modeling and Analysis Program
*
* revision 0.0 - 1/13/2010

* Qualilifiers for SET LIST: NCFORMAT,
* XCHUNK,YCHUNK,ZCHUNK,TCHUNK,DEFLATE,SHUFFLE,ENDIAN

	include	'ferret.parm'
	include	'slash.parm'
	include 'xcontext.cmn'
	include 'xprog_state.cmn'
	include 'errmsg.parm'
        include 'netcdf4_settings.cmn'
	include 'xrisc.cmn'          ! 12/94 SPLIT_LIST buffer

* calling argument declarations
	INTEGER		status

* internal variable declarations
	INTEGER  STR_SAME, loc, slen, nchunk
	CHARACTER TM_FMT*12, show_str*12
	REAL val

* initialize netcdf4-type stuff. If there has been a 
* SET LIST/NCFORMAT/ENDIAN/DEFLATE/SHUFFLE then these contain those values,
* otherwise they are set to the defaults


        risc_buff = '           ---- '
        CALL SPLIT_LIST(pttmode_explct, show_lun, risc_buff, 0)
        risc_buff = '        NetCDF-4 settings: '
        CALL SPLIT_LIST(pttmode_explct, show_lun, risc_buff, 0)

* /NCFORMAT for netCDF4 types ?
* NETCDF4 and CLASSIC, or 4=NETCDF4, 3=CLASSIC, 64BIT_OFFSET(?)

        IF (netcdf4_type .EQ. 3) 
     .      risc_buff = '        NetCDF file type = classic'
	IF (netcdf4_type .EQ. 4) 
     .      risc_buff = '        NetCDF file type = NetCDF-4'
	IF (netcdf4_type .EQ. 6) 
     .      risc_buff = '        NetCDF file type = NF_64BIT_OFFSET'
	CALL SPLIT_LIST(pttmode_explct, show_lun, risc_buff, 0)

* DEFLATE 
* Can set compression level though all indicatins are that value=1 is best
* deflate=0 means no deflate

        IF (deflate_lev .LE. 0) THEN
           risc_buff = 'Compression not set'
        ELSE 
	   WRITE (risc_buff, 1100) deflate_lev
	   CALL SPLIT_LIST(pttmode_explct, show_lun, risc_buff, 0)
	ENDIF
 1100   FORMAT (8X, 'Compression (deflate) level set to ', I1)

        nchunk = 0
        IF (xchunk_size_set .GT. 0) THEN
	   val = xchunk_size
           show_str = TM_FMT(val, 5, 12, slen)
	   WRITE (risc_buff, 1200) 'X', show_str(1:slen)
	   CALL SPLIT_LIST(pttmode_explct, show_lun, risc_buff, 0)
	   nchunk = nchunk + 1
	ENDIF
 1200   FORMAT (8X, A1,'CHUNK size set to ', A)
        IF (ychunk_size_set .GT. 0) THEN
	   val = ychunk_size
           show_str = TM_FMT(val, 5, 12, slen)
	   WRITE (risc_buff, 1200) 'Y', show_str(1:slen)
	   CALL SPLIT_LIST(pttmode_explct, show_lun, risc_buff, 0)
	   nchunk = nchunk + 1
	ENDIF

        IF (zchunk_size_set .GT. 0) THEN
	   val = zchunk_size
           show_str = TM_FMT(val, 5, 12, slen)
	   WRITE (risc_buff, 1200) 'Z', show_str(1:slen)
	   CALL SPLIT_LIST(pttmode_explct, show_lun, risc_buff, 0)
	   nchunk = nchunk + 1
	ENDIF

        IF (tchunk_size_set .GT. 0) THEN
	   val = tchunk_size
           show_str = TM_FMT(val, 5, 12, slen)
	   WRITE (risc_buff, 1200) 'T', show_str(1:slen)
	   CALL SPLIT_LIST(pttmode_explct, show_lun, risc_buff, 0)
	   nchunk = nchunk + 1
	ENDIF
        
	IF (nchunk .EQ. 0) THEN
	   risc_buff = '        NetCDF chunk sizes = default'
	   CALL SPLIT_LIST(pttmode_explct, show_lun, risc_buff, 0)
	ENDIF

* SHUFFLE 
* Shuffle = 0 or 1

	IF (shuffle_flag .EQ. 1) THEN
           risc_buff = '        SHUFFLE set to off'
	ELSE
           risc_buff = '        SHUFFLE set to on'
	ENDIF
	CALL SPLIT_LIST(pttmode_explct, show_lun, risc_buff, 0)

! ENDIAN: native, little, big

	IF (endian_code .EQ. 0) THEN
           risc_buff = '        ENDIAN set to native'
	ELSEIF (endian_code .EQ. 1) THEN
           risc_buff = '        ENDIAN set to little'
	ELSEIF (endian_code .EQ. 2) THEN
           risc_buff = '        ENDIAN set to big'
	ENDIF
	CALL SPLIT_LIST(pttmode_explct, show_lun, risc_buff, 0)

	RETURN

	END
