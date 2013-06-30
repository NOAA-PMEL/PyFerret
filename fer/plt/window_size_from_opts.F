        SUBROUTINE WINDOW_SIZE_FROM_OPTS(xinches, yinches, 
     .                    xpixels, ypixels, physical_chg, istat)

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
*  ARE DISCLAIMED. IN NO EVENT SHALL NOAA/PMEL/TMAP BE LIABLE FOR ANY 
*  SPECIAL, INDIRECT OR CONSEQUENTIAL DAMAGES OR ANY DAMAGES WHATSOEVER
*  RESULTING FROM LOSS OF USE, DATA OR PROFITS, WHETHER IN AN ACTION OF
*  CONTRACT, NEGLIGENCE OR OTHER TORTUOUS ACTION, ARISING OUT OF OR IN
*  CONNECTION WITH THE ACCESS, USE OR PERFORMANCE OF THIS SOFTWARE. 
*
*
* Deal with SET WINDOW size options.  If there are no size options
* specified, physical_change will be .FALSE.
*

        INCLUDE 'tmap_dims.parm'
        INCLUDE 'ferret.parm'
        INCLUDE 'errmsg.parm'
        INCLUDE 'plot_setup.parm'
        INCLUDE 'xplot_setup.cmn'
        INCLUDE 'xprog_state.cmn'

* calling argument declarations:
        REAL*4  xinches, yinches 
        INTEGER xpixels, ypixels, istat
        LOGICAL physical_chg

* function declaractions
        CHARACTER UPPER_CASE*1

* local parameter declarations:
        INTEGER slash_size,
     .          slash_aspect,
     .          slash_xpixels,
     .          slash_ypixels,
     .          slash_xinches,
     .          slash_yinches
        PARAMETER ( slash_size    =  0+1,
     .              slash_aspect  =  3+1,
     .              slash_xpixels = 13+1,
     .              slash_ypixels = 14+1,
     .              slash_xinches = 15+1,
     .              slash_yinches = 16+1 )

* internal variable declarations:
        LOGICAL   size_gvn, aspect_gvn, 
     .            xpixels_gvn, ypixels_gvn, 
     .            xinches_gvn, yinches_gvn
        INTEGER   qp, colon_pos
        REAL      val, aspect
        CHARACTER buffer*80

*       Set the default page size and image size
        xinches = dflt_xinches
        yinches = dflt_yinches
        xpixels = dflt_xpixels
        ypixels = dflt_ypixels
        physical_chg = .FALSE.

        aspect  = yinches / xinches

*       No size changes during animation
        IF ( animate ) 
     .     GOTO 4990

        size_gvn    = qual_given( slash_size )    .GT. 0
        aspect_gvn  = qual_given( slash_aspect )  .GT. 0
        xpixels_gvn = qual_given( slash_xpixels ) .GT. 0
        ypixels_gvn = qual_given( slash_ypixels ) .GT. 0
        xinches_gvn = qual_given( slash_xinches ) .GT. 0
        yinches_gvn = qual_given( slash_yinches ) .GT. 0

*       Check if there are any sizing options specified
        IF ( .NOT. (size_gvn .OR. aspect_gvn .OR. xpixels_gvn .OR.
     .              ypixels_gvn .OR. xinches_gvn .OR. yinches_gvn) ) 
     .     GOTO 4990

*       Check for conflicting sizing options
        IF ( aspect_gvn .AND. xpixels_gvn .AND. ypixels_gvn ) 
     .     GOTO 5010
        IF ( aspect_gvn .AND. xinches_gvn .AND. yinches_gvn ) 
     .     GOTO 5020
        IF ( xpixels_gvn .AND. ypixels_gvn .AND. 
     .       xinches_gvn .AND. yinches_gvn ) 
     .     GOTO 5030
        IF ( size_gvn .AND. (xpixels_gvn .OR. ypixels_gvn) )
     .     GOTO 5040

        physical_chg = .TRUE.

        IF ( aspect_gvn ) THEN
*          Get the given aspect ratio - edges only supported at this time
           qp = qual_given( slash_aspect )
           CALL EQUAL_VAL(cmnd_buff(qual_start(qp):qual_end(qp)),
     .                    aspect, istat)
           IF ( istat .NE. ferr_ok ) 
     .        RETURN
           IF ( aspect .EQ. unspecified_val4 ) THEN
              buffer = 'aspect ratio ?'
              GOTO 5050
           ENDIF
           IF ( aspect .LE. 0.0 ) THEN
              buffer = cmnd_buff(qual_start(qp):qual_end(qp))
              GOTO 5060
           ENDIF
        ENDIF

*       Get the page size
        IF ( xinches_gvn ) THEN
           qp = qual_given( slash_xinches )
           CALL EQUAL_VAL(cmnd_buff(qual_start(qp):qual_end(qp)),
     .                    val, istat)
           IF ( val .EQ. unspecified_val4 ) THEN
              buffer = 'xinches value ?'
              GOTO 5050
           ENDIF
           IF ( val .LE. 0.0 ) THEN
              buffer = cmnd_buff(qual_start(qp):qual_end(qp))
              GOTO 5060
           ENDIF
           xinches = val
        ENDIF
        IF ( yinches_gvn ) THEN
           qp = qual_given( slash_yinches )
           CALL EQUAL_VAL(cmnd_buff(qual_start(qp):qual_end(qp)),
     .                    val, istat)
           IF ( val .EQ. unspecified_val4 ) THEN
              buffer = 'yinches value ?'
              GOTO 5050
           ENDIF
           IF ( val .LE. 0.0 ) THEN
              buffer = cmnd_buff(qual_start(qp):qual_end(qp))
              GOTO 5060
           ENDIF
           yinches = val
        ENDIF
*       Get the aspect ratio if given by xinches and yinches
        IF ( xinches_gvn .AND. yinches_gvn )
     .     aspect = DBLE(yinches) / DBLE(xinches)

*       Get the image size
        IF ( size_gvn ) THEN
*          for backwards compatibility; not really recommended anymore
           qp = qual_given( slash_size )
           CALL EQUAL_VAL(cmnd_buff(qual_start(qp):qual_end(qp)),
     .                    val, istat)
           IF ( val .EQ. unspecified_val4 ) THEN
              buffer = 'size value ?'
              GOTO 5050
           ENDIF
           IF ( val .LE. 0.0 ) THEN
              buffer = cmnd_buff(qual_start(qp):qual_end(qp))
              GOTO 5060
           ENDIF
*          make it so the image scaling factor will be close to SQRT(val)
           xpixels = NINT(96.0 * dflt_xinches * SQRT(val))
           ypixels = NINT(96.0 * dflt_yinches * SQRT(val))
*          Aspect will be adjusted below
        ENDIF
        IF ( xpixels_gvn ) THEN
           qp = qual_given( slash_xpixels )
           CALL EQUAL_VAL(cmnd_buff(qual_start(qp):qual_end(qp)),
     .                    val, istat)
           IF ( val .EQ. unspecified_val4 ) THEN
              buffer = 'xpixels value ?'
              GOTO 5050
           ENDIF
           IF ( val .LE. 0.0 ) THEN
              buffer = cmnd_buff(qual_start(qp):qual_end(qp))
              GOTO 5060
           ENDIF
           xpixels = NINT(val)
        ENDIF
        IF ( ypixels_gvn ) THEN
           qp = qual_given( slash_ypixels )
           CALL EQUAL_VAL(cmnd_buff(qual_start(qp):qual_end(qp)),
     .                    val, istat)
           IF ( val .EQ. unspecified_val4 ) THEN
              buffer = 'ypixels value ?'
              GOTO 5050
           ENDIF
           IF ( val .LE. 0.0 ) THEN
              buffer = cmnd_buff(qual_start(qp):qual_end(qp))
              GOTO 5060
           ENDIF
           ypixels = NINT(val)
        ENDIF
*       Get the aspect ratio if given by xpixels and ypixels
        IF ( xpixels_gvn .AND. ypixels_gvn )
     .     aspect = DBLE(ypixels) / DBLE(xpixels)

*       Fill in missing xinches and yinches
        IF ( xinches_gvn .AND. .NOT. yinches_gvn ) THEN
           yinches = xinches * aspect
        ELSEIF ( yinches_gvn .AND. .NOT. xinches_gvn ) THEN 
           xinches = yinches / aspect
        ELSEIF ( .NOT. (xinches_gvn .OR. yinches_gvn) ) THEN
           xinches = SQRT(xinches * yinches / aspect)
           yinches = SQRT(xinches * yinches * aspect)
        ENDIF

*       Fill in missing xpixels and ypixels
        IF ( xpixels_gvn .AND. .NOT. ypixels_gvn ) THEN
           ypixels = NINT(xpixels * aspect)
        ELSEIF ( ypixels_gvn .AND. .NOT. xpixels_gvn ) THEN 
           xpixels = NINT(ypixels / aspect)
        ELSEIF ( .NOT. (xpixels_gvn .OR. ypixels_gvn) ) THEN
           xpixels = NINT(SQRT(xpixels * ypixels / aspect))
           ypixels = NINT(SQRT(xpixels * ypixels * aspect))
        ENDIF

* successful completion
 4990   istat = ferr_ok
 5000   RETURN

* error returns
 5010   CALL ERRMSG(ferr_invalid_command, istat, 
     .        'Only two of /ASPECT, /XPIXELS, and /YPIXELS '//
     .        'can be given', *5000)
 5020   CALL ERRMSG(ferr_invalid_command, istat, 
     .        'Only two of /ASPECT, /XINCHES, and /YINCHES '//
     .        'can be given', *5000)
 5030   CALL ERRMSG(ferr_invalid_command, istat, 
     .        'Only three of /XPIXLES, /YPIXELS, /XINCHES, '//
     .        'and /YINCHES can be given', *5000)
 5040   CALL ERRMSG(ferr_invalid_command, istat, 
     .        '/SIZE cannot be given along with '//
     .        'either /XPIXLES or /YPIXELS', *5000)
 5050   CALL ERRMSG(ferr_syntax, istat, buffer, *5000 )
 5060   CALL ERRMSG(ferr_out_of_range, istat, buffer, *5000)

        END

