	SUBROUTINE CDF_PREP_CACHE( dflt_cx, filename, nvars, memory,
     .			           mrlist, cxlist, clobber, status )

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
* setup to create a DODS NetCDF cache
* create a list of contexts appropriate to describe the (1,1,1,1) element
* of each variable in the DODS data set.

* programmer - steve hankin

* V450	11/96

#ifdef unix
        include 'tmap_errors.parm'
        include 'tmap_dims.parm'
        include 'xdset_info.cmn_text'
	include	'ferret.parm'
	include	'slash.parm'
	include 'errmsg.parm'
	include 'xvariables.cmn'
	include 'xcontext.cmn'
	include 'xprog_state.cmn'
	include 'xrisc_buff.cmn'
#else
        INCLUDE 'TMAP_FORMAT:TMAP_ERRORS.PARM'
	INCLUDE	'FERRET_CMN:FERRET.PARM'
	INCLUDE	'FERRET_CMN:SLASH.PARM'
	INCLUDE 'FERRET_CMN:ERRMSG.PARM'
        INCLUDE 'FERRET_CMN:GFDL_VMS.PARM'
	INCLUDE 'FERRET_CMN:XVARIABLES.CMN'
	INCLUDE 'FERRET_CMN:XCONTEXT.CMN'
	INCLUDE 'FERRET_CMN:XPROG_STATE.CMN'
	INCLUDE 'FERRET_CMN:XINTERRUPT.CMN'
#endif

* calling argument declarations:
	LOGICAL clobber
	INTEGER dflt_cx, nvars, mrlist(*), cxlist(*), status
	REAL	memory( mem_blk_size, max_mem_blks )
	CHARACTER TM_INQ_PATH*200, filename*(*)

* local variable declarations
	LOGICAL		TM_HAS_STRING, fexist
	INTEGER		TM_LENSTR1,
     .			qfile, dset, cx0, cx, ivar, idim, flen, dlen,
     .			iin, iout
	CHARACTER	ch*1, raw_name*200

* local parameter declarations
	INTEGER         slash_file
	PARAMETER     ( slash_file    = 12+1)

* determine the data set in question
        CALL GET_NEW_CX( dflt_cx, cx_cmnd, status )
        IF ( status .NE. ferr_ok ) GOTO 5000
	dset = cx_data_set(cx_cmnd)
	IF ( dset .EQ. unspecified_int4 ) GOTO 5100
	IF ( ds_type(dset) .NE. 'CDF' )   GOTO 5200
	IF ( .NOT.TM_HAS_STRING(ds_des_name(dset)(:8), 'http://') )
     .					  GOTO 5200

* determine the directory to receive the cached file
        qfile = qual_given( slash_file )
        IF ( qfile .GT. 0 ) THEN
           CALL EQUAL_STR_LC(
     .			cmnd_buff(qual_start(qfile):qual_end(qfile)),
     .                  raw_name, status )
           IF ( status .NE. ferr_ok ) GOTO 5000
* ... translate a possible env. var. at start of directory name.
*	Make sure the directory exists.
	   filename = TM_INQ_PATH( raw_name, ' ', ' ',
     .			.TRUE., status )
           IF ( status .NE. merr_ok ) GOTO 5000
	ELSE
	   filename = '.'
        ENDIF

* create the file name from the directory name
* ... name too long to cache?
	flen = TM_LENSTR1(filename)
	dlen = TM_LENSTR1(ds_des_name(dset))
	IF ( LEN(ds_des_name(dset)) .EQ. dlen ) GOTO 5300
	IF ( LEN(filename) .LE. flen+dlen-7 ) GOTO 5300
	filename(flen+1:flen+1) = '/'		! after user-supplied directory
	iout = flen+2				
	DO 10 iin = 8, dlen			! just past the "http://"
	   ch = ds_des_name(dset)(iin:iin)	
	   IF ( ch .EQ. '/' ) THEN
	      filename(iout:iout) = '_'
	   ELSE
	      filename(iout:iout) = ch 
	   ENDIF
	   iout = iout + 1
 10	CONTINUE

* if the file already exists and we are not in clobber mode ---> error
	IF ( .NOT.clobber ) THEN
	   INQUIRE( FILE = filename, EXIST = fexist )
	   IF ( fexist ) GOTO 5400
	ENDIF

* create a base context
        CALL STACK_PTR_UP( cx_stack_ptr, max_context, status )
        IF ( status .NE. ferr_ok ) GOTO 5000
        cx0 = cx_stack_ptr
        cx_data_set(cx0) = dset
        cx_category(cx0) = cat_file_var
        DO 20 idim = 1, 4
          cx_lo_ss(cx0,idim) = 1
          cx_hi_ss(cx0,idim) = 1
          cx_by_ss(idim,cx0) = .TRUE.
          cx_trans(idim,cx0) = trans_no_transform
          cx_given(idim,cx0) = .FALSE.
          cx_delta(idim,cx0) = unspecified_val8
 20    CONTINUE

* set up one dummy variable for each variable in the data set
	nvars = 0
	DO 100 ivar = 1, maxvars
	   IF ( ds_var_setnum(ivar) .NE. dset ) GOTO 100
           CALL STACK_PTR_UP( cx_stack_ptr, max_context, status )
           IF ( status .NE. ferr_ok ) GOTO 5000
           cx = cx_stack_ptr
	   nvars = nvars + 1
	   cxlist(nvars) = cx
	   CALL TRANSFER_CONTEXT(cx0, cx)
           cx_variable(cx) = ivar
	   cx_grid(cx)     = ds_grid_number(ivar)
	   cx_bad_data(cx) = ds_bad_flag( ivar )
	   DO 110 idim = 1, 4
	      CALL FLESH_OUT_AXIS( idim, cx, status )
	      IF ( status .NE. ferr_ok ) GOTO 5000
 110	   CONTINUE
	   CALL CREATE_TEMP_MEM_VAR( cx, mrlist(nvars), status )
	   IF ( status .NE. ferr_ok ) GOTO 5000
	   CALL UN_CACHE( mrlist(nvars) )
	   memory(1, mr_blk1(mrlist(nvars))) = 0.0    ! just good house-keeping
 100	CONTINUE

* echo the name of the output cache file
* 1/03 *kob* g77 port demands the use of risc_buff
	risc_buff = filename(1:TM_LENSTR1(filename))
	CALL SPLIT_LIST(pttmode_bull, ttout_lun,
     .		' LISTing to file '//risc_buff,0)

	RETURN

* error exits
 5000	CALL DELETE_MRS_IN_PROGRESS		! remove temporary vars
	RETURN
 5100	CALL ERRMSG( ferr_invalid_command, status,
     .			'no data set specified to cache', *5000 )
 5200	CALL ERRMSG( ferr_invalid_command, status,
     .			'can only cache a DODS NetCDF data set', *5000 )
 5300	CALL ERRMSG( ferr_invalid_command, status,
     .			'URL string too long to cache', *5000 )

* 1/03 *kob* g77 port demands the use of risc_buff
 5400	risc_buff = filename
	CALL ERRMSG( ferr_invalid_command, status,
     .			'/CLOBBER not specified and file exists: '
     .			//risc_buff, *5000 )

	END
