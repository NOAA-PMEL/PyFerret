      SUBROUTINE GET_MODE_ARG( mode, text, arg, status )

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
* interpret the text string accompanying the SET MODE command
*	eg. SET MODE TIME:HOURS

* programmer - steve hankin
* NOAA/PMEL, Seattle, WA - Tropical Modeling and Analysis Program
* written for VAX computer under VMS operating system
*
* revision 0.0 - 3/18/88
* V200:   6/8/89 - added mode desperate
*	 12/6/89 - and mode GKS
*	12/20/89 - and mode Xwindows
*	  1/5/89 - and mode metafile
* Unix/RISC port - 5/91 - cant use "//" with CHAR*(*) variable
*                - and X-windows names are more general
* V230:  5/14/92 - REMOTE_X mode was replaced by the REFRESH mode
*        11/9/92 - MODE META and MODE GKS no longer take arguments - eliminate
* V300:   2/3/93 - added MODE VERIFY FULL/FILE
*	 5/12/93 - file name args to SET MODE JOURNAL and SET MODE PPLLIST
* V301:  1/12/94 - allow filename arg for mode metafile
* V320 8/94 *kob* -IBM port - had to use risc buff for string concats
*       12/29/94 - eliminate unnecessary tty error message ("12/94")
*	 1/18/94 - added SET MODE VERIFY:ALWAYS (iarg=3)
*		   (bug fix for MODE VERIFY:garbage error msgs 2/3/95)
* V500: 4/99 *sh* - added MODE STUPID:weak_cache
* V570: *acm* 5/04 - add MODE GRATICULE[:argument] 
* V600: *acm* 5/06 - add new MODE LINECOLORS:[arg]
* V610  *acm* 3/08 - add new MODE NLEVELS:argument
* V67   *acm* 3/11 - Fix for ticket 1819, size of arg for MODE DESPERATE,
*                    and allow for a floating-point input
*       *acm* 3/12 6D Ferret (common uses nferdims in tmap_dims.parm)
* V698  *acm* 5/16 Ticket 2405: MODE CALENDAR:auto
* V702  *sh*  2/17 Added mode frugal
* v745 *acm* 12/18  Issue 1909, option to write dates in ISO8601 format
*                  new mode date_format, for formatted date output

        IMPLICIT NONE
	include 'tmap_dims.parm'
	include	'ferret.parm'
	include 'errmsg.parm'
	include 'gfdl_vms.parm'	
	include 'xprog_state.cmn'
	include 'xtext_info.cmn'
	include 'xrisc.cmn'          ! for RISC/FORTRAN

* calling argument declarations
	INTEGER		mode, arg, status
	REAL		farg
	CHARACTER*(*)	text

* local variable declarations:
	INTEGER	STR_CASE_BLIND_COMPARE, TM_LENSTR1,
     .		vax_code, tlen, cpos, iarg

* initialize
	status = ferr_ok
	IF ( text .EQ. ' ' ) THEN
	   IF ( mode .EQ. pmode_verify ) arg = 1   ! reset to "DEFAULT"
           IF ( mode .EQ. pmode_stupid ) arg = 2   ! same as unspecified_int4
!           IF ( mode .EQ. pmode_frugal ) arg = 30
	   RETURN
	ENDIF

* decode argument according to syntax for this mode
! removed 12/94
!	IF ( mode .EQ. pmode_diagnostic
!     .	.OR. mode .EQ. pmode_interpolate ) THEN
!	   WRITE ( ttout_lun, * ) 'mode argument not yet implemented'
!	ELSEIF ( mode .EQ. pmode_long_lab

	IF ( mode .EQ. pmode_long_lab
     .	    .OR. mode .EQ. pmode_lat_lab
     .	    .OR. mode .EQ. pmode_depth_lab ) THEN
	   READ ( text, '(BN,I10)', ERR=5010 ) iarg
	   IF ( ABS(iarg) .GT. 8  ) GOTO 5010
	   arg = iarg

	ELSEIF ( mode .EQ. pmode_time_lab ) THEN
	   tlen = MIN( 7, TM_LENSTR1(text) )
	   DO 100 iarg = 1, 7
	      vax_code = STR_CASE_BLIND_COMPARE
     .		( text(:tlen), date_labels(iarg)(:tlen) )
	      IF ( vax_code .EQ. vms_str_success ) THEN
	         arg = -iarg
	         GOTO 1000
	      ENDIF
 100	   CONTINUE
	   GOTO 5010	! unknown argument

	ELSEIF ( mode .EQ. pmode_desperate ) THEN
	   READ ( text, *, ERR=5010 ) farg
	   IF (farg .GT. 1.e9 .OR. farg .LE. 0) GOTO 5030
           iarg = INT(farg)
	   arg = iarg

	ELSEIF ( mode .EQ. pmode_verify ) THEN
	   tlen = MIN( 3, TM_LENSTR1(text) )
           DO 200 iarg = 1, 3
              vax_code = STR_CASE_BLIND_COMPARE
     .          ( text(1:3), verify_modes(iarg)(1:tlen) )
              IF ( vax_code .EQ. vms_str_success ) THEN
                 arg = iarg
                 GOTO 1000
              ENDIF
 200       CONTINUE
	   GOTO 5010
	ELSEIF ( mode .EQ. pmode_journal ) THEN
 	   journal_file = text
* ... change COMMON variable lttout in PLOT to output to file
	ELSEIF ( mode .EQ. pmode_ppllist ) THEN
	   ppllist_file = text
	ELSEIF ( mode .EQ. pmode_metafile ) THEN
	   CONTINUE     ! processed in XEQ_SET in case PLOT+ not opened yet
* ... SET MODE STUPID:weak_cache
	ELSEIF ( mode .EQ. pmode_stupid ) THEN
           vax_code = STR_CASE_BLIND_COMPARE(text(1:4), 'weak' )
           IF ( vax_code .EQ. vms_str_success ) THEN
              arg = 1
              GOTO 1000
           ELSE
	      GOTO 5010
           ENDIF
* ... SET MODE GRAT:graticule argument
	ELSEIF ( mode .EQ. pmode_grat ) THEN
            mode_grat_buff = text
            
	ELSEIF ( mode .EQ. pmode_linecolors ) THEN
	   READ ( text, '(BN,I10)', ERR=5010 ) iarg
	   IF ( iarg.LT.6 ) GOTO 5010
	   arg = iarg

	ELSEIF ( mode .EQ. pmode_nlevels ) THEN
	   READ ( text, '(BN,I10)', ERR=5020 ) iarg
	   IF ( iarg.LT.1 .OR. iarg .GT. 250 ) GOTO 5020
	   arg = iarg

	ELSEIF ( mode .EQ. pmode_frugal ) THEN
	   READ ( text, '(BN,I10)', ERR=5040 ) iarg
	   IF ( iarg.LT.0 .OR. iarg.GT.90) GOTO 5040
	   arg = iarg

* ... SET MODE datefmt:argument
*     if the argujent is ISO, then dates are written as ISO 8601 date strings

	ELSEIF ( mode .EQ. pmode_datefmt ) THEN
	   tlen = MIN( 3, TM_LENSTR1(text) )

           DO iarg = 1, 2
              vax_code = STR_CASE_BLIND_COMPARE
     .          ( text(1:3), datefmt_modes(iarg)(1:tlen) )
              IF ( vax_code .EQ. vms_str_success ) THEN
		 arg = pdate_vax
		 IF (datefmt_modes(iarg)(1:3).EQ.'ISO') arg = pdate_numslashnum
                 GOTO 1000
              ENDIF
           ENDDO
	   GOTO 5700

        ELSE
	   GOTO 5010	! mode does not accept argument
	ENDIF

* successful completion
 1000	RETURN
        
* error exit
 5000   RETURN
 5010   tlen = TM_LENSTR1(text)
        risc_buff = text(:tlen)
        CALL ERRMSG( ferr_out_of_range, status,
     .   'illegal mode argument '//risc_buff, *5500 )
 5020   tlen = TM_LENSTR1(text)
        risc_buff = text(:tlen)
        CALL ERRMSG( ferr_out_of_range, status,
     .   'illegal mode argument. Must be in [1,250] '//risc_buff, *5500 )
 5030   tlen = TM_LENSTR1(text)
        risc_buff = text(:tlen)
        CALL ERRMSG( ferr_out_of_range, status,
     .   'illegal mode argument. Negative or > 1.e+9, ', *5500 )
 5040   CALL ERRMSG( ferr_out_of_range, status,
     .   'illegal mode argument. Must be 0 to 90 percent '
     .          //text(:tlen), *5000 )
 5500   IF ( mode .EQ. pmode_verify ) CALL WARN(
     .          'Legal VERIFY arguments are DEFAULT, ALL, and ALWAYS')
 5700   IF ( mode .EQ. pmode_datefmt ) CALL WARN(
     .          'Legal DATEFMT arguments are DEFAULT and ISO8601')
        GOTO 5000

        END

