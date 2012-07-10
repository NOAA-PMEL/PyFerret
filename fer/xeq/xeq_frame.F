	SUBROUTINE XEQ_FRAME

*
*
*  This software was developed by the Thermal Modeling and Analysis
*  Project(TMAP) of the National Oceanographic and Atmospheric
*  Administration''s (NOAA) Pacific Marine Environmental Lab(PMEL),
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
* PyFr  *kms* 11/11 changes for saving using the graphics delegate
*       *acm* 3/12 6D Ferret (common uses nferdims in tmap_dims.parm)
* V6.8 *acm* 7/9/2012 For batch graphics on RHEL6, change ITS_BATCH_GRAPHICS
*                     from logical to integer, then batch = ITS_BATCH_GRAPHICS().NE.0

        IMPLICIT NONE
	include 'tmap_dims.parm'
	include 'ferret.parm'
	include 'errmsg.parm'
	include 'slash.parm'
	include 'xprog_state.cmn'
	include 'xplot_state.cmn'
	include 'xplot_setup.cmn'

* internal function return value declarations:
        INTEGER   TM_LENSTR1

* internal variable declarations:
        INTEGER   sts, loc, flen
        CHARACTER fformat*5, filename*1028

* decode the qualifiers, if any

* FRAME/TRANSPARENT

        transparent = 0
        loc = qual_given( slash_frame_transp )
        IF (loc .GT. 0 .AND. .NOT. animate) THEN
           transparent = 1
        ENDIF

* FRAME/FORMAT=...
        loc = qual_given( slash_frame_format )
        IF ( loc .GT. 0 ) THEN
           CALL EQUAL_STR_LC(
     .          cmnd_buff(qual_start(loc):qual_end(loc)),
     .                    fformat, sts )
           IF ( sts .NE. ferr_ok ) GOTO 5000
        ELSE
           fformat = ''
        ENDIF

* FRAME/FILE=filename
        loc = qual_given( slash_frame_file )
        IF ( loc .GT. 0 ) THEN
           CALL EQUAL_STR_LC(
     .          cmnd_buff(qual_start(loc):qual_end(loc)),
     .                    filename, sts )
           IF ( sts .NE. ferr_ok ) GOTO 5000
        ELSEIF ( fformat .NE. '' ) THEN
           filename = 'ferret.' // fformat
        ELSE
           filename = frame_file        ! saved SET MOVIE/FILE name
        ENDIF
        flen = TM_LENSTR1(filename)

* rename previous versions of this file using a version number scheme
        CALL TM_PARK_LAST_VERSION(filename(:flen), sts)

* save that baby
        CALL SAVE_FRAME(filename(:flen), fformat, sts)

        RETURN

* error exits
 5000   RETURN
        END

