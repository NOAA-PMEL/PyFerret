	SUBROUTINE OPEN_SHOW_FILE ( lun, sho_file,  
     .                             clobber, append, status)


	
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
* OPEN the output file for SHOW xx/OUTFILE=  or MESSAGE/OUTFILE= 

* programmer - Ansley Manke TMAP
* 3/2010
*       *acm* 3/12 6D Ferret (common uses nferdims in tmap_dims.parm)

        IMPLICIT NONE
	include	'tmap_dims.parm'
	include	'ferret.parm'
	include	'slash.parm'
	include 'errmsg.parm'
	include 'xprog_state.cmn'

* Argument declarations:
	LOGICAL clobber, append
	INTEGER sho_file, lun, status

* local parameter declarations:
	LOGICAL fexist
	INTEGER TM_LENSTR1, slen, flen
	CHARACTER filename*2048, raw_name*2048, fformatted*12,
     .            fstatus*8, faccess*12, fposition*12

* set up the output device

	fformatted = 'FORMATTED'
* ... get full file name - from SHOW/OUTFILE= or from SET SHOW/OUTFILE
	raw_name = ' '

* ... sho_file is from qual_given(slash_show_axis_file) or
*     qual_given(slash_show_data_file) etc.

* preserve case of filename

	CALL EQUAL_STR_LC(
     .			cmnd_buff(qual_start(sho_file):qual_end(sho_file)),
     .                  raw_name, status )

	IF ( status .NE. ferr_ok ) GOTO 5000
	IF ( raw_name .EQ. ' ') THEN

* ... use default from SET SHOW/outfile= command
	   raw_name = show_file
	   IF ( raw_name .EQ. ' ') GOTO 5100
	ENDIF

	slen = TM_LENSTR1(raw_name)
	IF (slen .GT. 256 ) GOTO 5200
	INQUIRE( FILE = raw_name, NAME = filename, EXIST = fexist )

	flen = TM_LENSTR1( filename )

	IF ( fexist ) THEN    
	   IF ( append ) THEN
	      faccess = 'APPEND'
	      fposition = 'APPEND'
	      fstatus = 'OLD'
	   ELSEIF ( clobber ) THEN
	      faccess = 'SEQUENTIAL'
	      fstatus = 'REPLACE'
	   ELSE
	      GOTO 5050	 !attempt to clobber a file
	   ENDIF
	ELSE
	   faccess = 'SEQUENTIAL'
	   fstatus = 'NEW'
	   fposition = 'asis'
	ENDIF
* ... open or create it

	OPEN( UNIT	= default_list_lun,
     .	      FILE	= raw_name,
     .	      FORM	= fformatted,
     .	      ACCESS	= faccess,
     .	      STATUS	= fstatus,
     .	      ERR		= 5010 )

	lun = default_list_lun

* error exit
 5000	RETURN
 
 5010	CALL ERRMSG( ferr_erreq,status,filename(:flen),*5000 )
 5100	CALL ERRMSG( ferr_invalid_command, status,
     .		'SHOW /FILE= what name?', *5000 )
 5050	CALL ERRMSG( ferr_invalid_command, status,
     .			'/CLOBBER not specified and file exists: '
     .			//raw_name, *5000 )
 5200	CALL ERRMSG( ferr_prog_limit, status,
     .  'Cannot write to file w/ name longer than 256 characters: '//pCR//
     .  'Limitation of Fortran INQUIRE and OPEN calls '	
     .			, *5000 )

	END

