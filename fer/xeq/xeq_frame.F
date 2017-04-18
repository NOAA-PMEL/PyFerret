        SUBROUTINE XEQ_FRAME(memory)

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

* FRAME/FILM        - to trigger actual movie frame
* FRAME/BITMAP        - to store the current workstation screen contents

* programmer - steve hankin
* NOAA/PMEL, Seattle, WA - Tropical Modeling and Analysis Program
* written for VAX computer under VMS operating system
*
* revision 0.0 - 5/7/87
* V200:  11/14/89 - guts moved to SAVE_FRAME - laser output, too
* V320: 5/94 - array "memory" as a calling argument
*         8/94 - added qualifiers /FORMAT and /FRAME to permit direct capture
*                of GIF images
* V320: 1/95 - use gif_frame routine for both HDF and GIF
*        call TM_PARK_LAST_VERSION for names
* V500: 4/99 *sh* increased filename string length from 128 to 255
* V602: *acm* 1/07 Changes for metafile-batch mode: Ignore FRAME in 
*                  batch mode, except when in batch GIF mode.
* V6.2  *acm* 5/08 new qualifier FRAME/TRANSPARENT
* PyFr  *kms* 11/11 changes for saving using the graphics delegate
*       *acm* 3/12 6D Ferret (common uses nferdims in tmap_dims.parm)
* V6.8 *acm* 7/9/2012 For batch graphics on RHEL6, change ITS_BATCH_GRAPHICS
*                     from logical to integer, then batch = ITS_BATCH_GRAPHICS().NE.0
* PyFr  *kms*  7/13 Add /XPIXELS, /YPIXELS, /XINCHES, /YINCHES to FRAME
* PyFr  *kms*  8/13 Add /ANNOTATE to FRAME; declare memory argument 
*                   now needed for SAVE_FRAME
* The following changes were not applied to PyFerret:
* V6.95 *acm* 4/15 Haven't used HDF images for a long time: require GIF only.

        IMPLICIT NONE
        include 'tmap_dims.parm'
        include 'ferret.parm'
        include 'errmsg.parm'
        include 'slash.parm'
        include 'xprog_state.cmn'
        include 'xplot_state.cmn'
        include 'xplot_setup.cmn'
        include 'gkscm1_inc.decl'
        include 'GKSCM1.INC'      ! wsid

        REAL memory(*)

* internal function return value declarations:
        INTEGER   TM_LENSTR1

* internal variable declarations:
        REAL      val
        REAL*4    xinches, yinches
        INTEGER   sts, loc, flen, xloc, yloc,
     .            xpixels, ypixels
        CHARACTER annovar*1028, fformat*5, filename*1028

* decode the qualifiers, if any

* FRAME /XINCHES=... xor /YINCHES=...
        xloc = qual_given( slash_frame_xinches )
        yloc = qual_given( slash_frame_yinches )
        IF ( (xloc .GT. 0) .AND. (yloc .GT. 0) ) THEN
            GOTO 5010
        ELSEIF ( xloc .GT. 0 ) THEN
            CALL EQUAL_VAL(cmnd_buff(qual_start(xloc):qual_end(xloc)), 
     .                     val, sts)
            IF ( sts .NE. ferr_ok ) RETURN
            IF ( sts .EQ. unspecified_val4 ) GOTO 5020
            IF ( val .LT. 1.25 ) GOTO 5030
            xinches = val
            yinches = val * wn_yinches(wsid) / wn_xinches(wsid)
        ELSEIF ( yloc .GT. 0 ) THEN
            CALL EQUAL_VAL(cmnd_buff(qual_start(yloc):qual_end(yloc)), 
     .                     val, sts)
            IF ( sts .NE. ferr_ok ) RETURN
            IF ( sts .EQ. unspecified_val4 ) GOTO 5040
            IF ( val .LT. 1.25 ) GOTO 5050
            yinches = val
            xinches = val * wn_xinches(wsid) / wn_yinches(wsid)
        ELSE
            xinches = wn_xinches(wsid)
            yinches = wn_yinches(wsid)
        ENDIF

* FRAME /XPIXELS=... xor /YPIXELS=...
        xloc = qual_given( slash_frame_xpixels )
        yloc = qual_given( slash_frame_ypixels )
        IF ( (xloc .GT. 0) .AND. (yloc .GT. 0) ) THEN
            GOTO 5015
        ELSEIF ( xloc .GT. 0 ) THEN
            CALL EQUAL_VAL(cmnd_buff(qual_start(xloc):qual_end(xloc)), 
     .                     val, sts)
            IF ( sts .NE. ferr_ok ) RETURN
            IF ( sts .EQ. unspecified_val4 ) GOTO 5025
            IF ( val .LT. 128 ) GOTO 5035
            xpixels = NINT(val)
            ypixels = NINT(val * wn_ypixels(wsid) / wn_xpixels(wsid))
        ELSEIF ( yloc .GT. 0 ) THEN
            CALL EQUAL_VAL(cmnd_buff(qual_start(yloc):qual_end(yloc)), 
     .                     val, sts)
            IF ( sts .NE. ferr_ok ) RETURN
            IF ( sts .EQ. unspecified_val4 ) GOTO 5045
            IF ( val .LT. 128 ) GOTO 5055
            ypixels = NINT(val)
            xpixels = NINT(val * wn_xpixels(wsid) / wn_ypixels(wsid))
        ELSE
            xpixels = NINT(wn_xpixels(wsid))
            ypixels = NINT(wn_ypixels(wsid))
        ENDIF

* FRAME/ANNOTATE=...
        loc = qual_given( slash_frame_annotate )
        IF ( loc .GT. 0 ) THEN
            CALL EQUAL_STR_LC(
     .              cmnd_buff(qual_start(loc):qual_end(loc)), 
     .              annovar, sts)
            IF ( sts .NE. ferr_ok ) RETURN
            IF ( annovar .EQ. ' ' ) GOTO 5060
        ELSE
            annovar = ' '
        ENDIF

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
        CALL SAVE_FRAME(filename(:flen), fformat, annovar,
     .                  xinches, yinches, xpixels, ypixels, sts)

 5000   RETURN

* error exits
 5010   CALL ERRMSG( ferr_invalid_command, sts,
     .       'Only one of /XINCHES and /YINCHES can be given', *5000 )
 5015   CALL ERRMSG( ferr_invalid_command, sts,
     .       'Only one of /XPIXELS and /YPIXELS can be given', *5000 )
 5020   CALL ERRMSG( ferr_syntax, sts, '/XINCHES=?', *5000 )
 5025   CALL ERRMSG( ferr_syntax, sts, '/XPIXELS=?', *5000 )
 5030   CALL ERRMSG( ferr_out_of_range, sts, 
     .       '/XINCHES must be at least 1.25', *5000)
 5035   CALL ERRMSG( ferr_out_of_range, sts, 
     .       '/XPIXELS must be at least 128', *5000)
 5040   CALL ERRMSG( ferr_syntax, sts, '/YINCHES=?', *5000 )
 5045   CALL ERRMSG( ferr_syntax, sts, '/YPIXELS=?', *5000 )
 5050   CALL ERRMSG( ferr_out_of_range, sts, 
     .       '/YINCHES must be at least 1.25', *5000)
 5055   CALL ERRMSG( ferr_out_of_range, sts, 
     .       '/YPIXELS must be at least 128', *5000)
 5060   CALL ERRMSG( ferr_syntax, sts, '/ANNOTATE=?', *5000 )

        END

