	SUBROUTINE XEQ_FRAME

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

#ifdef unix
	include 'ferret_cmn/ferret.parm'
	include 'ferret_cmn/errmsg.parm'
	include 'ferret_cmn/slash.parm'
	include 'ferret_cmn/xprog_state.cmn'
	include 'ferret_cmn/xplot_state.cmn'
#else
	INCLUDE 'FERRET_CMN:FERRET.PARM'
	INCLUDE 'FERRET_CMN:ERRMSG.PARM'
	INCLUDE 'FERRET_CMN:SLASH.PARM'
	INCLUDE 'FERRET_CMN:XPROG_STATE.CMN'
	INCLUDE 'FERRET_CMN:XPLOT_STATE.CMN'
#endif

* internal variable declarations:
	LOGICAL		is_format, is_file
	INTEGER		TM_LENSTR1, status, loc, flen
	CHARACTER	format*3, filename*128

* initialize
	is_format = qual_given( slash_frame_format ).GT.0
	is_file   = qual_given( slash_frame_file   ).GT.0

* decode the qualifiers, if any
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

