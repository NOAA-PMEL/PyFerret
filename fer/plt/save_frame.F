        SUBROUTINE SAVE_FRAME(filename, fformat, annovar,
     .                        xinches, yinches, xpixels, ypixels, sts)
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
* PyFr  *kms*  8/13 Add /ANNOTATE to FRAME; include memory argument
* PyFr  *kms*  4/17 Revised for new memory management

* calling argument declarations:
        IMPLICIT NONE
        CHARACTER*(*) filename, fformat, annovar
        REAL*4 xinches, yinches
        INTEGER xpixels, ypixels, sts

* function return value declarations
        INTEGER TM_LENSTR

        INCLUDE 'tmap_dims.parm'
        INCLUDE 'ferret.parm'
        INCLUDE 'errmsg.parm'
        INCLUDE 'xcontext.cmn'
        INCLUDE 'xplot_state.cmn'
        INCLUDE 'xprog_state.cmn'
        INCLUDE 'xvariables.cmn'
        INCLUDE 'fgrdel.cmn'

* internal variable declarations:
        TYPE(mem_table_slot) :: firststr
        INTEGER windowid, namlen, fmtlen, success, errstrlen, 
     .          mr, cx, k, memlo, memhi, steplo, stephi, numstr
        CHARACTER*2048 errstr

        namlen = TM_LENSTR(filename)
        fmtlen = TM_LENSTR(fformat)

* determine the current output window (GKS wkstn ID, that is)
        DO 10 windowid = 1, max_windows
           IF ( wn_active(windowid) ) GOTO 110
   10   CONTINUE

*       Sanity checks
  110   IF ( (windowid .LT. 1) .OR. 
     .       (windowid .GT. maxwindowobjs) ) THEN
            errstr = 'SAVE_FRAME: invalid windowid'
            errstrlen = TM_LENSTR(errstr)
            CALL SPLIT_LIST(pttmode_help, err_lun, errstr, errstrlen)
            sts = FERR_INVALID_COMMAND
            RETURN
        ENDIF
        IF ( windowobjs(windowid) .EQ. nullobj ) THEN
            errstr = 'SAVE_FRAME: null windowobj'
            errstrlen = TM_LENSTR(errstr)
            CALL SPLIT_LIST(pttmode_help, err_lun, errstr, errstrlen)
            sts = FERR_INVALID_COMMAND
            RETURN
        ENDIF

*       Get the annotation strings to add at the top
        IF ( annovar .NE. ' ' ) THEN
*           Get the string array given by the variable named in annovar
            CALL GET_FER_COMMAND('LOAD ' // annovar, sts, *5200)
            IF ( sts .NE. FERR_OK ) GOTO 5200
*           Get the string data into memory
            CALL GET_CMND_DATA(cx_last, ptype_string, sts)
            IF ( sts .NE. FERR_OK ) GOTO 5200
            mr = is_mr(isp)
            cx = is_cx(isp)

*           Starting place in memory for this array of pointers to C strings
            firststr = memry(mr)
*           If the whole array was not specified, a new copy of the data
*           has been made in memory with unit increments (or so it appears).
            numstr = 0
            DO 120 k = 1,nferdims
                memlo = mr_lo_ss(mr,k)
                memhi = mr_hi_ss(mr,k)
                steplo = cx_lo_ss(cx,k)
                stephi = cx_hi_ss(cx,k)
                IF ( (memlo .NE. unspecified_int4) .OR.
     .               (memhi .NE. unspecified_int4) ) THEN
                   IF ( (steplo .NE. memlo) .OR.
     .                  (stephi .NE. memhi) ) GOTO 5400
                   IF ( numstr .EQ. 0 ) THEN
                       numstr = stephi - steplo + 1
                   ELSE
                       numstr = numstr * (stephi - steplo + 1)
                   ENDIF
                ENDIF
  120       CONTINUE
            IF ( numstr .LE. 0 ) GOTO 5400
        ELSE
            numstr = 0
*           firststr not used by fgdwinsave
        ENDIF

        CALL FGDWINSAVE(success, windowobjs(windowid), filename,
     .                  namlen, fformat, fmtlen, transparent, 
     .                  xinches, yinches, xpixels, ypixels,
     .                  firststr, numstr)
        IF ( success .EQ. 0 ) GOTO 5100

* successful completion
        sts = ferr_ok
 5000   RETURN

* error exit
 5100   errstr = ' '
        CALL FGDERRMSG(errstr, errstrlen)
        CALL ERRMSG(ferr_invalid_command, sts, errstr, *5000)
 5200   errstr = 'problems evaluating ' // annovar
        CALL ERRMSG(ferr_invalid_command, sts, errstr, *5000)
 5400   errstr = annovar//' is not a valid string array'
        CALL ERRMSG(ferr_invalid_command, sts, errstr, *5000)
 
        END

