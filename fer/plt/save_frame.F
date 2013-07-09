        SUBROUTINE SAVE_FRAME(filename, fformat, xinches, yinches, 
     .                        xpixels, ypixels, sts)
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
* make a movie or GIF frame of the current output graphics

* programmer - steve hankin
* NOAA/PMEL, Seattle, WA - Tropical Modeling and Analysis Program
* written for VAX computer under VMS operating system
*
* V200:  11/14/89 - based on XEQ_FRAME
* V320: 5/94 - array "memory" as a calling argument
*       8/94 - filename added as calling argument
*       1/95 - major revamp using entirely X code instead of GKS (much faster)
*              and incorporating GIF output
*       9/97 - now supports batch mode
* V500: 4/99 *sh* increased filename string length from 128 to 255
* 1/03 *kob* g77 port - requires use of risc_buff for concats
*
* V6.2  *jli* 5/08 pass background color and transparent to put_frame_batch()
* PyFr  *kms* 11/11 major rewrite using graphics delegate function call
*       *acm* 3/12 6D Ferret (common uses nferdims in tmap_dims.parm)
* V6.8 *acm* 7/9/2012 For batch graphics on RHEL6, change ITS_BATCH_GRAPHICS
*                     from logical to integer, then batch = ITS_BATCH_GRAPHICS().NE.0
* PyFr  *kms*  7/13 Add /XPIXELS, /YPIXELS, /XINCHES, /YINCHES to FRAME

* calling argument declarations:
        IMPLICIT NONE
        CHARACTER*(*) filename, fformat
        REAL*4 xinches, yinches
        INTEGER xpixels, ypixels, sts

* function return value declarations
        INTEGER TM_LENSTR

* internal variable declarations:
        INTEGER windowid, namlen, fmtlen, success, errstrlen
        CHARACTER*2048 errstr

        include 'errmsg.parm'
        include 'xplot_state.cmn'
        include 'fgrdel.cmn'

        namlen = TM_LENSTR(filename)
        fmtlen = TM_LENSTR(fformat)

* determine the current output window (GKS wkstn ID, that is)
        DO 10 windowid = 1, max_windows
           IF ( wn_active(windowid) ) GOTO 110
   10   CONTINUE

*       Sanity checks
  110   IF ( (windowid .LT. 1) .OR. 
     .       (windowid .GT. maxwindowobjs) ) THEN
            STOP 'SAVE_FRAME: Invalid windowid value'
        ENDIF
        IF ( windowobjs(windowid) .EQ. nullobj ) THEN
            STOP 'SAVE_FRAME: null windowobj'
        ENDIF

        CALL FGDWINSAVE(success, windowobjs(windowid), filename,
     .                  namlen, fformat, fmtlen, transparent, 
     .                  xinches, yinches, xpixels, ypixels)
        IF ( success .EQ. 0 ) GOTO 5100

* successful completion
        sts = ferr_ok
 5000   RETURN

* error exit
 5100   errstr = ' '
        CALL FGDERRMSG(errstr, errstrlen)
        CALL ERRMSG(ferr_invalid_command, sts, errstr, *5000)

        END

