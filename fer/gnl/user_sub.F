	SUBROUTINE USER_SUB( memory,
     .			     dat1, dat2, dat3, dat4, dat5, dat6,
     .			     mr_list, cx_list, nvars, lun, status )

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
* background information
* Sample information about the variables that have been passed is extracted
* for you below by way of example
* The qualifiers /COMMAND and /FILE are also extracted.  Using /COMMAND=name
* allows you to select among many calculations to be performed by this routine.
* Also 2 general purpose qualifiers, /OPT1 and /OPT2 are provided.

* programmer - steve hankin
* NOAA/PMEL, Seattle, WA - Tropical Modeling and Analysis Program
* written for VAX computer under VMS operating system

* V200:	 1/21/90
* V301: 11/93 - added objective analysis:
*	 USER/COMMAND="objective analysis"
*	     /OPT1=xlo:xhi:xdel,ylo:yhi:ydel,[cay],[rng]
*	     /FILE=filename
*       1/94 added /FORMAT=(FORTRAN format)
*	1/94 increased the number of args passed to 6
*	2/94 added /COMMAND=SAMPLE
* V312: 5/94 - array "memory" as a calling argument
* V320: 12/29/94 - use SPLIT_LIST to redirect tty output for GUI
* 12/96 *kob* - Linux Port - had to move variable assignments off of
*               declaration line.

#ifdef unix
	include 'ferret_cmn/ferret.parm'
	include 'ferret_cmn/errmsg.parm'
	include 'ferret_cmn/xvariables.cmn'
	include 'ferret_cmn/xcontext.cmn'
	include 'ferret_cmn/xprog_state.cmn'
	include	'ferret_cmn/xmem_subsc.cmn'
#else
	INCLUDE 'FERRET_CMN:FERRET.PARM'
	INCLUDE 'FERRET_CMN:ERRMSG.PARM'
	INCLUDE 'FERRET_CMN:XVARIABLES.CMN'
	INCLUDE 'FERRET_CMN:XCONTEXT.CMN'
	INCLUDE 'FERRET_CMN:XPROG_STATE.CMN'
	INCLUDE	'FERRET_CMN:XMEM_SUBSC.CMN'
#endif

* calling argument declarations:
	INTEGER	nvars, mr_list(nvars), cx_list(nvars), lun, status
	REAL	memory(*),
     .		dat1( m1lox:m1hix,m1loy:m1hiy,m1loz:m1hiz,m1lot:m1hit ),
     .		dat2( m2lox:m2hix,m2loy:m2hiy,m2loz:m2hiz,m2lot:m2hit ),
     .		dat3( m3lox:m3hix,m3loy:m3hiy,m3loz:m3hiz,m3lot:m3hit ),
     .		dat4( m4lox:m4hix,m4loy:m4hiy,m4loz:m4hiz,m4lot:m4hit ),
     .		dat5( m5lox:m5hix,m5loy:m5hiy,m5loz:m5hiz,m5lot:m5hit ),
     .		dat6( m6lox:m6hix,m6loy:m6hiy,m6loz:m6hiz,m6lot:m6hit )

* internal variable declarations:
	LOGICAL	TM_HAS_STRING, opt1, opt2, binary, file_given
	INTEGER	TM_LENSTR1,
     .		grid(6), lo_lim(4,6), hi_lim(4,6),
     .		i, j, k, l, ngood, nbad, cx, mr, slen, idim, loc, ivar,
     .		len_line, len_set_name, columns
	REAL	bad_flag(6), datum
	CHARACTER	VAR_TITLE*128, VAR_TITLE_MOD*128, AX_TITLE*32,
     .			LEFINT*6
	CHARACTER	vname_buff*128,	line_buff*130, title*32,
     .			opt1_string*80, opt2_string*80, filename*128,
     .			fmt*80, command*40, bl12*12
	PARAMETER	(bl12 = '            ' )

* local parameter declarations
	INTEGER	slash_opt1, slash_opt2, slash_command,
     .		slash_file, slash_format
	PARAMETER ( slash_opt1 = 1,
     .		    slash_opt2 = 2,
     .		    slash_command = 3,
     .		    slash_file = 13,
     .		    slash_format = 14 )

* initialize
	opt1_string = ' '
	opt2_string = ' '
	command = ' '
	filename = ' '
	fmt = ' '
	binary = .FALSE.
	columns = 0

* DECODE THE COMMAND QUALIFIERS:
* use the "command" qualifer to consolidate multiple commands under USER
* use the "opt1" and "opt2" qualifiers as you see fit
*  (logical variables opt1 and opt2 at your disposal)
* ... /COMMAND= (case insensitive)
	loc  = qual_given( slash_command )
	IF ( loc .GT. 0 ) THEN
	   CALL EQUAL_STRING( cmnd_buff(qual_start(loc):qual_end(loc)),
     .			      command, status )
           IF ( status .NE. ferr_ok ) RETURN
	ENDIF
* ... /OPT1
	loc  = qual_given( slash_opt1 )
	opt1 = loc .GT. 0
	IF ( opt1 ) THEN
	   CALL EQUAL_STRING( cmnd_buff(qual_start(loc):qual_end(loc)),
     .			      opt1_string, status )
           IF ( status .NE. ferr_ok ) RETURN
	ENDIF
* ... /OPT2
	loc  = qual_given( slash_opt2 )
	opt2 = loc .GT. 0
	IF ( opt2 ) THEN
	   CALL EQUAL_STRING( cmnd_buff(qual_start(loc):qual_end(loc)),
     .			      opt2_string, status )
           IF ( status .NE. ferr_ok ) RETURN
	ENDIF
* ... /FILE=  (case sensitive)
	loc  = qual_given( slash_file )
	file_given  = loc .GT. 0
	IF ( file_given ) THEN
	   CALL EQUAL_STR_LC( cmnd_buff(qual_start(loc):qual_end(loc)),
     .			      filename, status )
           IF ( status .NE. ferr_ok ) RETURN
	ENDIF
* ... /FORMAT=(FORTRAN format)
	loc  = qual_given( slash_format )
	IF ( loc .GT. 0 ) THEN
	   CALL EQUAL_STRING( cmnd_buff(qual_start(loc):qual_end(loc)),
     .			      fmt, status )
           IF ( status .NE. ferr_ok ) RETURN
	   binary = fmt.EQ.' ' .OR. TM_HAS_STRING(fmt,'unf')
	   IF ( binary ) THEN     ! this is kludge, buggy, expedient logic ...
	      loc = INDEX(fmt,':')  ! e.g. format=unf:201
	      IF ( loc .GT. 0 ) THEN
#ifdef MANDATORY_FORMAT_WIDTHS
	         READ (fmt(loc+1:), *) columns
#else
	         READ (fmt(loc+1:),'(I)') columns
#endif
	         IF ( columns .LT. 0 ) columns = 0	         
	      ENDIF
	      fmt = ' '
	   ELSE
	      CALL CHECK_FORMAT( fmt, status ) 
              IF ( status .NE. ferr_ok ) RETURN
	   ENDIF
	ENDIF

* extract other values that may be needed: grids, bad_flags, limits
	DO 110 ivar = 1, nvars
	   mr = mr_list(ivar)
	   cx = cx_list(ivar)
	   grid(ivar)		= mr_grid( mr )
	   bad_flag(ivar)	= mr_bad_data( mr )
	   DO 100 idim = 1, 4
	      lo_lim(idim,ivar) = cx_lo_ss(cx,idim)
	      hi_lim(idim,ivar) = cx_hi_ss(cx,idim)
 100	   CONTINUE
 110	CONTINUE

* !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!*
****************** MODIFY FROM HERE DOWN ***************************

* -----------------------------------------------------------------
* 11/93 addition for (semi-supported) objective analysis (and sampling 2/94)
* /COMMAND="objective analysis" (may be abbreviated to "obj")
	IF ( TM_HAS_STRING(command, 'OBJ') ) THEN
	   CALL OBJECTIVE( memory, dat1, dat2, dat3,
     .			   mr_list, cx_list, nvars,
     .			   opt1_string, opt2_string, filename,
     .			   fmt, status )
	   RETURN
* /COMMAND="SAMPLE" (may be abbreviated to "SAMP")
	ELSEIF ( TM_HAS_STRING(command, 'SAMP') ) THEN
	   CALL SAMPLE(    memory, mr_list, cx_list, nvars,
     .			   opt1_string, opt2_string, file_given,
     .			   filename, fmt, binary, columns, status )
	   RETURN
	ENDIF
* -----------------------------------------------------------------

* output background on the data
	DO 210 ivar = 1, nvars
	   mr = mr_list(ivar)
	   cx = cx_list(ivar)
* ... first the variable name
	   vname_buff   = VAR_TITLE( cx )
	   CALL SPLIT_LIST(pttmode_explct, lun,' ',1)
	   CALL SPLIT_LIST(pttmode_explct, lun,
     .		bl12//vname_buff(1:TM_LENSTR1(vname_buff)), 0)

* ... qualify title with modification line ?
	   vname_buff = VAR_TITLE_MOD( cx )
	   IF ( vname_buff .NE. ' ' ) THEN
	      len_line   = TM_LENSTR1( vname_buff )
	      CALL SPLIT_LIST(pttmode_explct, lun,
     .		bl12//vname_buff(1:len_line), 0)
	   ENDIF

* ... then location information
	   DO 200 idim = 1,4
	      CALL ASCII_LIMITS( cx, idim, line_buff, len_line )
	      title = AX_TITLE( idim, grid, slen )
	      CALL SPLIT_LIST(pttmode_explct, lun,
     .		bl12//title( :slen )//': '//line_buff(:len_line),0)
 200	   CONTINUE

* ... then data set name or names
	   CALL GET_DSET_NAME( cx_data_set(cx), line_buff, len_set_name )
	   CALL SPLIT_LIST(pttmode_explct, lun,
     .		bl12//'DATA SET: '// line_buff(:len_set_name), 0)
 210	CONTINUE

* EXAMPLE OF A CALCULATION
	nbad  = 0
	ngood = 0
	DO 500 l = lo_lim(4,1), hi_lim(4,1)
	DO 500 k = lo_lim(3,1), hi_lim(3,1)
	DO 500 j = lo_lim(2,1), hi_lim(2,1)
	DO 500 i = lo_lim(1,1), hi_lim(1,1)

	   datum = dat1(i,j,k,l)

	   IF ( datum .EQ. bad_flag(1) ) THEN
	      nbad = nbad + 1
	   ELSE
	      ngood = ngood + 1
	   ENDIF
 500	CONTINUE

* display results
	CALL SPLIT_LIST(pttmode_explct, lun,' ',1)
	CALL SPLIT_LIST(pttmode_explct, lun,
     .		'Total # of data points: '//LEFINT(nbad+ngood,slen),0)
	CALL SPLIT_LIST(pttmode_explct, lun,
     .		'# flagged as bad  data: '//LEFINT(nbad,slen),0)
	RETURN
	END

