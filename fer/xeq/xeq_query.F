    	SUBROUTINE XEQ_QUERY

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
* execute the QUERY command

* programmer - steve hankin
* NOAA/PMEL, Seattle, WA - Tropical Modeling and Analysis Program

* for FERRET V300: 6/16/93
* V301:	11/05/93 - changed QUERY/PIPE to QUERY/IGNORE (a NO-OP)
*Linux Port 1/97 *kob* - Added a preprocessor include for tmap_dset.parm
*			  because it needed a preprocessor.
* V66   *kms* 5/10 LIB_GET_LUN is a subroutine, not a function
*       *acm* 3/12 cleanup ifdefs and unnecessary include files
** 10/2022 *acm* Code cleanup: remove references to obsolete gui interface

        IMPLICIT NONE
        include 'tmap_dims.parm'
#       include "tmap_dset.parm"
        include 'xdset_info.cmn_text'
        external xdset_info_data
        include 'xtm_grid.cmn_text'
        external xgt_grid_data
	include	'ferret.parm'
	include	'slash.parm'
	include 'errmsg.parm'
	include 'xprog_state.cmn'
	include 'xcommand.cmn'

* local parameter declarations:
	INTEGER	    slash_all, slash_file, slash_ignore
	PARAMETER ( slash_all  = 1 + 0,
     .		    slash_file = 1 + 1,
     .		    slash_ignore = 1 + 2 )
	CHARACTER*1 NO_TAG
	PARAMETER ( NO_TAG = " " )

* local variable declarations:
	LOGICAL	do_all, fileout, its_ppl
	INTEGER	TM_LENSTR1, STR_UPCASE, vax_code,
     .		chout, flen, qfile, status, i, query, n
	CHARACTER fname*132, qry4*4, q_fmt1*20

	DATA q_fmt1 / '(I3, 1X, A4, 1x, A)' /

* initialize
	fileout = .FALSE.
	status = ferr_ok
        qfile = qual_given( slash_file )

* is this a QUERY/IGNORE NO-OP ?
	IF ( qual_given( slash_ignore ) .GT. 0 ) RETURN

* query given ?
	IF ( num_args .EQ. 0 ) GOTO 5300

* command given with /ALL ?
	do_all = qual_given( slash_all ) .GT. 0

* identify the query - if it starts with 'P' it is a PLOT+ query
	qry4 = cmnd_buff(arg_start(1):arg_end(1))
	vax_code = STR_UPCASE( qry4,
     .		   cmnd_buff(arg_start(1):arg_end(1)) )
	its_ppl = qry4(1:1) .EQ. 'P'
	IF ( .NOT. its_ppl ) THEN
	   DO 10 query = 1, max_queries
	      IF ( queries(query) .EQ. unspecified_name4 ) GOTO 10
	      IF ( qry4 .EQ. queries(query)(:4) )          GOTO 20
 10	   CONTINUE
	   GOTO 5100
	ENDIF

* command given with /FILE  - open output channel
 20        IF ( qfile .GT. 0 ) THEN
	   CALL EQUAL_STR_LC(
     .                  cmnd_buff(qual_start(qfile):qual_end(qfile)),
     .                  fname, status )
	   IF ( status .NE. ferr_ok ) GOTO 5000
	   flen = TM_LENSTR1(fname)
	   IF ( fname.NE.' ' ) THEN
	      CALL LIB_GET_LUN( chout ) 
	      OPEN ( UNIT = chout,
     .		     FILE = fname,
     .		     STATUS = 'UNKNOWN',
     .		     ERR = 5200 ) 
	      fileout = .TRUE.
	   ENDIF

	ELSE
	   chout = ttout_lun
	ENDIF

* do it
	IF ( .NOT. its_ppl ) CALL DO_QUERY( chout, q_fmt1, query, NO_TAG, status )
	IF ( status .NE. ferr_ok ) GOTO 5000
	
* exit
 1000	IF ( fileout ) THEN
	   WRITE ( chout, '(40A1)' ) ('=',i=1,40) 
	   CLOSE ( UNIT = chout, ERR = 5000 )
	ENDIF
	RETURN
 3000	FORMAT (I4, 2X, A4)

* error exits
 5000	RETURN
 5100	CALL ERRMSG( ferr_invalid_command, status,
     .		     'unknown QUERY: '//cmnd_buff(arg_start(1):arg_end(1)),
     .		     *1000 )
 5200	CALL ERRMSG( ferr_erreq, status, fname(1:flen), *1000 )
 5300	CALL ERRMSG( ferr_invalid_command, status, 'QUERY what?', *5000 )
	END

