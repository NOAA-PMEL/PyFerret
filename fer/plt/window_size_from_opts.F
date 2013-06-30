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

        IMPLICIT NONE
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
*       the following are given below
        REAL HEIGHT_FROM_AXIS_ASPECT, 
     .       WIDTH_FROM_AXIS_ASPECT

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
        REAL      val, aspect, xtmp, ytmp
        CHARACTER buffer*80, numbuff*24, asp_ref*1

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

*       Default to edges for the aspect
        asp_ref = 'E'
        IF ( aspect_gvn ) THEN
* ... decode the ratio and possible ":SIZE" or ":AXES" qualifier
           qp = qual_given( slash_aspect )
           CALL EQUAL_STRING(cmnd_buff(qual_start(qp):qual_end(qp)),
     .                       buffer, istat)
           IF ( istat .NE. ferr_ok ) RETURN
           IF ( istat .EQ. unspecified_val4 ) THEN
              buffer = 'aspect ratio ?'
              GOTO 5060
           ENDIF
           colon_pos = INDEX(buffer, ':')
           IF ( colon_pos .GT. 0 ) THEN
              numbuff = buffer(:colon_pos-1)
              READ (numbuff, *, ERR=5050) aspect
              asp_ref = UPPER_CASE(buffer(colon_pos+1:colon_pos+1))
              IF ( (asp_ref .NE. 'A') .AND. (asp_ref .NE. 'E') ) 
     .           GOTO 5050
           ELSE
              READ (buffer, *, ERR=5050) aspect
           ENDIF
           IF ( aspect .LE. 0.0 ) GOTO 5070
        ENDIF

*       Get the page size
        IF ( xinches_gvn ) THEN
           qp = qual_given( slash_xinches )
           CALL EQUAL_VAL(cmnd_buff(qual_start(qp):qual_end(qp)),
     .                    val, istat)
           IF ( istat .NE. ferr_ok ) RETURN
           IF ( val .EQ. unspecified_val4 ) THEN
              buffer = 'xinches value ?'
              GOTO 5060
           ENDIF
           IF ( val .LE. 0.0 ) GOTO 5070
           xinches = val
        ENDIF
        IF ( yinches_gvn ) THEN
           qp = qual_given( slash_yinches )
           CALL EQUAL_VAL(cmnd_buff(qual_start(qp):qual_end(qp)),
     .                    val, istat)
           IF ( istat .NE. ferr_ok ) RETURN
           IF ( val .EQ. unspecified_val4 ) THEN
              buffer = 'yinches value ?'
              GOTO 5060
           ENDIF
           IF ( val .LE. 0.0 ) GOTO 5070
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
           IF ( istat .NE. ferr_ok ) RETURN
           IF ( val .EQ. unspecified_val4 ) THEN
              buffer = 'size value ?'
              GOTO 5060
           ENDIF
           IF ( val .LE. 0.0 ) GOTO 5070
*          make it so the image scaling factor will be close to SQRT(val)
           xpixels = NINT(96.0 * dflt_xinches * SQRT(val))
           ypixels = NINT(96.0 * dflt_yinches * SQRT(val))
*          Aspect will be adjusted below
        ENDIF
        IF ( xpixels_gvn ) THEN
           qp = qual_given( slash_xpixels )
           CALL EQUAL_VAL(cmnd_buff(qual_start(qp):qual_end(qp)),
     .                    val, istat)
           IF ( istat .NE. ferr_ok ) RETURN
           IF ( val .EQ. unspecified_val4 ) THEN
              buffer = 'xpixels value ?'
              GOTO 5060
           ENDIF
           IF ( val .LE. 0.0 ) GOTO 5070
           xpixels = NINT(val)
        ENDIF
        IF ( ypixels_gvn ) THEN
           qp = qual_given( slash_ypixels )
           CALL EQUAL_VAL(cmnd_buff(qual_start(qp):qual_end(qp)),
     .                    val, istat)
           IF ( istat .NE. ferr_ok ) RETURN
           IF ( val .EQ. unspecified_val4 ) THEN
              buffer = 'ypixels value ?'
              GOTO 5060
           ENDIF
           IF ( val .LE. 0.0 ) GOTO 5070
           ypixels = NINT(val)
        ENDIF
*       Get the aspect ratio if given by xpixels and ypixels
        IF ( xpixels_gvn .AND. ypixels_gvn )
     .     aspect = DBLE(ypixels) / DBLE(xpixels)

*       Fill in missing xinches and yinches
        IF ( xinches_gvn .AND. .NOT. yinches_gvn ) THEN
           IF ( asp_ref .EQ. 'E' ) THEN
              yinches = xinches * aspect
           ELSE
              yinches = HEIGHT_FROM_AXIS_ASPECT(xinches, aspect)
           ENDIF
        ELSEIF ( yinches_gvn .AND. .NOT. xinches_gvn ) THEN 
           IF ( asp_ref .EQ. 'E' ) THEN
              xinches = yinches / aspect
           ELSE
              xinches = WIDTH_FROM_AXIS_ASPECT(yinches, aspect)
           ENDIF
        ELSEIF ( .NOT. (xinches_gvn .OR. yinches_gvn) ) THEN
           IF ( asp_ref .EQ. 'E' ) THEN
              xtmp = SQRT(xinches * yinches / aspect)
              ytmp = SQRT(xinches * yinches * aspect)
              xinches = xtmp
              yinches = ytmp
           ELSE
              xtmp = WIDTH_FROM_AXIS_ASPECT(yinches, aspect)
              ytmp = HEIGHT_FROM_AXIS_ASPECT(xinches, aspect)
*             Using the geometric means keeps the page area 
*             the same and preserves the aspect ratio
              xinches = SQRT(xtmp * xinches)
              yinches = SQRT(yinches * ytmp)
           ENDIF
        ENDIF

*       Change an axis aspect ratio to an edges aspect ratio
        IF ( asp_ref .EQ. 'A' ) THEN
           aspect = yinches / xinches
           asp_ref = 'E'
        ENDIF

*       Fill in missing xpixels and ypixels
        IF ( xpixels_gvn .AND. .NOT. ypixels_gvn ) THEN
           ypixels = NINT(xpixels * aspect)
        ELSEIF ( ypixels_gvn .AND. .NOT. xpixels_gvn ) THEN 
           xpixels = NINT(ypixels / aspect)
        ELSEIF ( .NOT. (xpixels_gvn .OR. ypixels_gvn) ) THEN
           xtmp = SQRT(xpixels * ypixels / aspect)
           ytmp = SQRT(xpixels * ypixels * aspect)
           xpixels = NINT(xtmp)
           ypixels = NINT(ytmp)
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
 5050   buffer = cmnd_buff(qual_start(qp):qual_end(qp))
 5060   CALL ERRMSG(ferr_syntax, istat, buffer, *5000 )
 5070   buffer = cmnd_buff(qual_start(qp):qual_end(qp))
        CALL ERRMSG(ferr_out_of_range, istat, buffer, *5000)

        END

***********************************************************************
*       The following solve the equation:
*          (h - hm1 - hm2) / (w - wm1 -wm2) = asp
*       for h given w, or for w given h,
*       where hm1, hm2, wm1, wm2 are the scaled standard margins
*       and the scaling is SQRT(w * h / (std_w * std_h))
*       This reduces down to solving:
*          h^2 - h * w * (V + 2 * asp) + w^2 * asp^2 = 0
*       for h or w, where V is a term of only standard height, width, and margins
***********************************************************************

        REAL FUNCTION HEIGHT_FROM_AXIS_ASPECT(xin, asp)

        IMPLICIT NONE
        INCLUDE 'plot_setup.parm'

* calling argument declarations:
        REAL*4 xin
        REAL asp

* local argument declarations:
        REAL val, sqval, rootp, rootm, yout,
     .       sc, xax, yax, aspp, aspm

        val = ( dflt_ywhitelo + dflt_ywhitehi
     .          - asp * dflt_xwhitelo - asp * dflt_xwhitehi )
        val = val * val
     .        / (dflt_xinches * dflt_yinches)
        val = val + 2.0 * asp
        sqval = val * val - 4.0 * asp * asp
*       .GT. instead of .NE. allows for some floating point precision error
        IF ( sqval .GT. 0.0 ) THEN
*          figure out which root to use
*          if margins all the same, then rootp for asp < 1, rootm for asp > 1
           rootp = val + SQRT(sqval)
           yout = 0.5 * xin * rootp
           sc = SQRT(xin * yout / (dflt_xinches * dflt_yinches))
           xax = xin - sc * (dflt_xwhitelo + dflt_xwhitehi)
           yax = yout - sc * (dflt_ywhitelo + dflt_ywhitehi)
           aspp = yax / xax

           rootm = val - SQRT(sqval)
           yout = 0.5 * xin * rootm
           sc = SQRT(xin * yout / (dflt_xinches * dflt_yinches))
           xax = xin - sc * (dflt_xwhitelo + dflt_xwhitehi)
           yax = yout - sc * (dflt_ywhitelo + dflt_ywhitehi)
           aspm = yax / xax

           IF ( ABS(aspp - asp) .LT. ABS(aspm - asp) ) THEN
              val = rootp
           ELSE
              val = rootm
           ENDIF
        ENDIF

        yout = 0.5 * xin * val
        HEIGHT_FROM_AXIS_ASPECT = yout
        RETURN
        END

***********************************************************************

        REAL FUNCTION WIDTH_FROM_AXIS_ASPECT(yin, asp)

        IMPLICIT NONE
        INCLUDE 'plot_setup.parm'

* calling argument declarations:
        REAL*4 yin
        REAL asp

* local argument declarations:
        REAL val, sqval, rootp, rootm, xout,
     .       sc, xax, yax, aspp, aspm

        val = ( dflt_ywhitelo + dflt_ywhitehi
     .          - asp * dflt_xwhitelo - asp * dflt_xwhitehi )
        val = val * val
     .        / (dflt_xinches * dflt_yinches)
        val = val + 2.0 * asp
        sqval = val * val - 4.0 * asp * asp
*       .GT. instead of .NE. allows for some floating point precision error
        IF ( sqval .GT. 0.0 ) THEN
*          figure out which root to use
*          if margins all the same, then rootp for asp > 1, rootm for asp < 1
           rootp = val + SQRT(sqval)
           xout = 0.5 * yin * rootp / (asp * asp)
           sc = SQRT(xout * yin / (dflt_xinches * dflt_yinches))
           xax = xout - sc * (dflt_xwhitelo + dflt_xwhitehi)
           yax = yin - sc * (dflt_ywhitelo + dflt_ywhitehi)
           aspp = yax / xax

           rootm = val - SQRT(sqval)
           xout = 0.5 * yin * rootm / (asp * asp)
           sc = SQRT(xout * yin / (dflt_xinches * dflt_yinches))
           xax = xout - sc * (dflt_xwhitelo + dflt_xwhitehi)
           yax = yin - sc * (dflt_ywhitelo + dflt_ywhitehi)
           aspm = yax / xax

           IF ( ABS(aspp - asp) .LT. ABS(aspm - asp) ) THEN
              val = rootp
           ELSE
              val = rootm
           ENDIF
        ENDIF

        xout = 0.5 * yin * val / (asp * asp)
        WIDTH_FROM_AXIS_ASPECT = xout
        RETURN
        END

