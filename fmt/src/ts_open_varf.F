	SUBROUTINE TS_OPEN_VARF ( vf_num,status )

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
* Open a var file. If the validity of the file hasn't been checked yet
* check its firt 4 bytes as an indication of the right file
* type. If the file doesn't exist return status but DONT call ERRMSG

* programmer - Mark Verschell
* NOAA/PMEL, Seattle, WA - Tropical Modeling and Analysis Program
* written for VAX computer under VMS operating system
*
* revision 0.00 - 08/23/88
* revision 0.10 - 1/91 *sh* syntax changes to accomodate DEC RISC compiler
* Unix/RISC port - 3/19/91 *sh*: file record length computed and stored in
*                  vf_reclen during ts_init_dset
*                - 4/24/91: use TM_INQ_PATH to locate file
*                         : and check for matching machine fp formats
* SUN port - 1/30/92 -kob- 1) Moved 'external xmachine_data out of ifdef so it 
*                              will become valid on the vax.
*                          2) xmachine_byte.cmn on SUN and VAX
*                             and xmachine_int1.cmn on DECstation because
*                             because it understands INTEGER*1
* replaced "elif" syntax with
*	else
*	   if
*  for SGI port	 - kob 4/8/92
*
* IBM port 14apr94 *kob* : readonly not allowed in open statments, so I
*			 : added an "ifdef" check for READONLY_ALLOWED
* arguments:
*	vf_num - "var_file" number (in COMMON/XSTEP_FILE/) of file to open
*	status	- (output) error flag 

#ifdef unix
	include 'tmap_format/tmap_errors.parm'	! error codes
#include "tmap_format/tmap_dset.parm"
	include 'tmap_format/tmap_dims.parm'
	include 'tmap_format/xstep_files.cmn_text'
	external xstep_files_data
	include 'tmap_format/xdset_info.cmn_text'
	external xdset_info_data
	include 'tmap_format/xio.cmn_text'
	external xio_data
        include 'tmap_format/xsys_names.cmn_text'
        external xsys_names_data
#ifdef sun
      include 'tmap_format/xmachine_byte.cmn'
#else
      include 'tmap_format/xmachine_int1.cmn'
#endif

#else
	INCLUDE 'TMAP_FORMAT:TMAP_ERRORS.PARM'	! error codes
	INCLUDE 'TMAP_FORMAT:TMAP_DSET.PARM'
	INCLUDE 'TMAP_FORMAT:TMAP_DIMS.PARM'
	INCLUDE 'TMAP_FORMAT:XSTEP_FILES.CMN'
	INCLUDE 'TMAP_FORMAT:XDSET_INFO.CMN'
	INCLUDE 'TMAP_FORMAT:XIO.CMN'
        INCLUDE 'TMAP_FORMAT:XMACHINE_BYTE.CMN'
        INCLUDE 'TMAP_FORMAT:XSYS_NAMES.CMN'
#endif

* moved out of ifdef unix to become valid on vax as well 1/92 -kob-
        EXTERNAL XMACHINE_DATA

	INTEGER*4	vf_num,lunit,status, nrecpervar           
        INTEGER*2       npospervar
	CHARACTER*4	rec_code,nskip_char,var_code
	CHARACTER*13	TM_STRING
        CHARACTER*128   TM_INQ_PATH, var_file
#ifdef sun
        BYTE            author_cpu
#else
#	if unix  
        	INTEGER*1       author_cpu
#	else
        	BYTE            author_cpu
#	endif
#endif

* initialize: get file characteristics
* *sh* 3/91: removed RECL = vf_reclen(vf_num)
* *sh* 4/91: path/logical name (tmap_tsteps) not stored in descriptor file
	IF ( .NOT. vf_valid(vf_num) ) THEN
           var_file = TM_INQ_PATH( vf_name(vf_num), tmap_tsteps, ' ',
     .                             .TRUE., status )
           IF ( status .NE. merr_ok ) GOTO 9010
           vf_name(vf_num) = var_file     ! save exact path
	ENDIF

* get free logical unit #
	CALL LIB_GET_LUN (lunit)

	OPEN   (UNIT = lunit,
     .		FILE = vf_name(vf_num),
     .		RECL = vf_reclen(vf_num),
     .		FORM = 'UNFORMATTED',
     .		ACCESS = 'DIRECT',
     .		STATUS = 'OLD',
#ifdef READONLY_ALLOWED
     .		READONLY,
#endif
     .		ERR = 5000)

* we're done if validity was already checked
	IF (vf_valid(vf_num)) GOTO 1000

* ----------- check validity of varfile --------
* check to see if the first records should be skipped
	READ (UNIT=lunit, REC=1, ERR=5000) rec_code,nskip_char
	IF (rec_code .EQ. 'SKIP') THEN
	  READ (nskip_char, FMT='(I4)', ERR=5000) vf_skip(vf_num)
	ELSE
	  vf_skip(vf_num) = 0
	ENDIF

* check validity of first VAR record
	READ (UNIT=lunit, REC=vf_skip(vf_num)+1, ERR=5000)
     .			rec_code,var_code,nrecpervar,npospervar,author_cpu
	IF (rec_code .NE. 'VAR ') CALL TM_ERRMSG
     .		(merr_varform, status, 'TS_OPEN_VARF',
     .		 vf_setnum(vf_num), vf_num,
     .		 'VAR='//ds_var_code(vf_firstvar(vf_num)),
     .		 rec_code//' x_coord='//TM_STRING(var_code), *9000)
	IF (ds_var_code(vf_firstvar(vf_num)) .NE. var_code)
     .	   CALL TM_ERRMSG (merr_varmtch, status, 'TS_OPEN_VARF',
     .		           vf_setnum(vf_num), vf_num,
     .	                   ds_var_code(vf_firstvar(vf_num)),
     .	                   var_code, *9000)

* should check var_code for correct positioning (analagous to gt_open_step)
*   ?????

* need to convert binary fp if written on a different machine ?
        vf_cpu_type(vf_num) = author_cpu

* successful completion - leave varfile open
1000	status = merr_ok
	vf_valid(vf_num) = .TRUE.
	vf_lunit(vf_num) = lunit
	GOTO 9010

* error exit - close file
 5000	CALL TM_ERRMSG (merr_erreq, status, 'TS_OPEN_VARF',
     .		 	vf_setnum(vf_num), vf_num,
     .			no_errstring, no_errstring, *9000)
 9000	CLOSE (UNIT = lunit, ERR=9010)		! ERR= in case file wasn't open
 9010	RETURN
	END
