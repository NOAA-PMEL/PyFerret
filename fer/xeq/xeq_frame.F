	SUBROUTINE XEQ_FRAME

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
* execute the FRAME command
* make a movie frame of the current output graphics

* FRAME/FILM	- to trigger actual movie frame
* FRAME/BITMAP	- to store the current workstation screen contents

* programmer - steve hankin
* NOAA/PMEL, Seattle, WA - Tropical Modeling and Analysis Program
* written for VAX computer under VMS operating system
*
* revision 0.0 - 5/7/87
* V200:  11/14/89 - guts moved to SAVE_FRAME - laser output, too
* V320: 5/94 - array "memory" as a calling argument
* 	8/94 - added qualifiers /FORMAT and /FRAME to permit direct capture
*		of GIF images
* V320: 1/95 - use gif_frame routine for both HDF and GIF
*	call TM_PARK_LAST_VERSION for names
* V500: 4/99 *sh* increased filename string length from 128 to 255
* V602: *acm* 1/07 Changes for metafile-batch mode: Ignore FRAME in 
*                  batch mode, except when in batch GIF mode.
* V6.2  *acm* 5/08 new qualifier FRAME/TRANSPARENT[=color]

	include 'ferret.parm'
	include 'errmsg.parm'
	include 'slash.parm'
	include 'xprog_state.cmn'
	include 'xplot_state.cmn'
	include 'xplot_setup.cmn'

* internal variable declarations:
	LOGICAL		ITS_GIF_GRAPHICS, ITS_BATCH_GRAPHICS,
     .                  is_format, is_file, its_gif, batch
	INTEGER		TM_LENSTR1, STR_SAME, status, loc, flen, qp
	CHARACTER	format*3, filename*255

* initialize
	is_format = qual_given( slash_frame_format ).GT.0
	is_file   = qual_given( slash_frame_file   ).GT.0

* Ignore FRAME in batch mode, except when in batch GIF mode.

	its_gif = ITS_GIF_GRAPHICS()
	batch = ITS_BATCH_GRAPHICS()

        IF (batch .AND. .NOT.its_gif) RETURN

* decode the qualifiers, if any

* FRAME/TRANSPARENT[=color]

	transparent = 0
	qp = qual_given( slash_frame_transp )
	IF (qp .GT. 0 .AND. .NOT. animate) THEN
	   IF ( .NOT. batch ) CALL WARN 
     . ('ignoring FRAME/TRANSPARENT. It takes effect only in GIF mode')
           transparent = 1
           CALL EQUAL_STR_LC( cmnd_buff(qual_start(qp):qual_end(qp)),
     .                transparent_color, status )
           IF (STR_SAME(transparent_color, ' ') .EQ. 0)
     .                transparent_color = 'white'
        ENDIF

* FRAME/FORMAT="HDF" or "GIF"
	IF ( is_format ) THEN
	   loc = qual_given( slash_frame_format )
	   CALL EQUAL_STRING(
     .          cmnd_buff(qual_start(loc):qual_end(loc)),
     .                    format, status )
	   IF ( status .NE. ferr_ok ) GOTO 5000
	   IF ( format .NE. 'GIF' .AND. format .NE. 'HDF' ) GOTO 5100
	ELSE
	   format = 'HDF'
	ENDIF

* FRAME/FILE=filename
	IF ( is_file ) THEN
	   loc = qual_given( slash_frame_file )
	   CALL EQUAL_STR_LC(
     .          cmnd_buff(qual_start(loc):qual_end(loc)),
     .                    filename, status )
	   IF ( status .NE. ferr_ok ) GOTO 5000
	ELSEIF ( format .EQ. 'GIF' ) THEN
	   filename = 'ferret.gif'
	ELSE
	   filename = frame_file	! saved SET MOVIE/FILE name
	ENDIF
	flen = TM_LENSTR1(filename)

* a format of "GIF" can be implied by a filename ending in .gif or .GIF
* any other name will implicitly be an HDF file (unless explicit w/ /FORM=)
	IF ( .NOT.is_format .AND. flen .GT. 4) THEN
	   IF ( filename(flen-3:flen) .EQ. '.gif'
     .     .OR. filename(flen-3:flen) .EQ. '.GIF' )  format = 'GIF'
	ENDIF

* rename previous versions of this file using a version number scheme
	IF ( format .EQ. 'GIF' )  CALL TM_PARK_LAST_VERSION
     .			( filename(:flen), status )

* save that baby
        CALL SAVE_FRAME(filename(:flen), format, status)

	RETURN

* error exits
 5000	RETURN
 5100	CALL ERRMSG( ferr_invalid_command, status,
     .               'illegal format type - must be "HDF" or "GIF"'
     .		     //pCR//cmnd_buff(:len_cmnd), *5000 )
	END

