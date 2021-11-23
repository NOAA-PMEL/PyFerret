	SUBROUTINE XEQ_PPLUS

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
* access program PPLUS ( the underlying graphics package ) directly - either
* by sending a single command to it or by entering the PPLUS command interpreter

* programmer - steve hankin
* NOAA/PMEL, Seattle, WA - Tropical Modeling and Analysis Program
* written for VAX computer under VMS operating system
*
* revision 0.0 - 1/7/87
* revision 0.1 - 9/8/88 - added /RESET option
* revision 0.2 - 1/31/89 - added PPLUS interruptibility
* Unix/RISC port - 10/9/91 - check to see if user changes ylen or yorg
* V500 *sh* 6/99 - for commands %LABEL and %WHERE define PLOT+ symbols
*			mode verify output, and journal file comments
*       *acm* 3/12 cleanup ifdefs and unnecessary include files
* V6.85 *acm* 3/13 Fix ticket 2049; check we can open palette file
* PyFr  *kms*  7/13 Reset scaling with PPL /RESET
* V6.86 *acm* 1/14 In above fix, let there be spaces before the file spec.
* V7.4 *acm* 1/18 for issue 1854 if mem pointers for curvi coord data have
*                 been temporarily reset, restore to prev values
* V764 11/21 *acm* For PyFerret issue 98, if PLOT/SET/PALETTE, save info so the color 
*                 palette can be restored after the plot is completed. Here restore the palette.
 
        IMPLICIT NONE
	include 'tmap_dims.parm'
	include	'ferret.parm'
	include 'errmsg.parm'
	include	'plot_setup.parm'
	include 'xprog_state.cmn'
	include 'xplot_state.cmn'
	include 'xplot_setup.cmn'
	include 'xgrinpt.cmn'	 ! mouse input
	include 'xvariables.cmn'
        include 'axis_inc.decl'
        include 'AXIS.INC'       ! ylen,yorg
        include 'plt_inc.decl'
        include 'PLT.INC'        ! width, height
	include 'gkscm1_inc.decl'
	include 'GKSCM1.INC'	 ! wsid
        INCLUDE 'fgrdel.cmn'     ! windowdpix, windowdpiy
	INCLUDE   'parampl5_dat.decl'

* local variable declarations:
	LOGICAL	ver_out, jnl_out
	INTEGER TM_LENSTR1, STR_UPCASE, i, slen, start, status
        REAL temp_ylen, temp_yorg
	CHARACTER*16 vbuff
	CHARACTER TM_INQ_PATH*2048, fname*128, sfile*2048
        REAL*4 scl

* local parameter declarations:
	INTEGER		slash_reset
	PARAMETER     ( slash_reset = 0+1 )

* turn on PLOT+ ?
	CALL START_PPLUS(.FALSE.)

* save ylen and yorg in case the user manually changes those values in PLOT+
* 10/9/91 addition 
        temp_ylen = ylen
        temp_yorg = yorg

* Check for a PPL SHASET SPECTRUM command, and verify the palette file can be opened.
	IF ( num_args .GT. 0 ) THEN
           IF ( INDEX(cmnd_buff(arg_start(1):len_cmnd),'SPECTRUM') .GT. 0 ) THEN
	      start = INDEX(cmnd_buff,'SPECTRUM=') + 10
              fname = cmnd_buff(start:len_cmnd)
              CALL TM_PACK_STRING( fname, fname, 1, 128, slen )
	      slen = TM_LENSTR1(fname)
              sfile = TM_INQ_PATH (fname(:slen),'FER_PALETTE','.spk',
     .        	 .false.,status)
              IF (status .NE. ferr_ok) go to 5300 
           ENDIF

           IF ( INDEX(cmnd_buff(arg_start(1):len_cmnd),'PATTERN') .GT. 0 ) THEN
	      start = INDEX(cmnd_buff,'PATTERN=') + 9
              fname = cmnd_buff(start:len_cmnd)
              CALL TM_PACK_STRING( fname, fname, 1, 128, slen )
	      slen = TM_LENSTR1(fname)
              sfile = TM_INQ_PATH (fname(:slen),'FER_PALETTE','.pat',
     .        	 .false.,status)
              IF (status .NE. ferr_ok) go to 5300 
           ENDIF

        ENDIF

	IF ( qual_given( slash_reset ) .GT. 0 ) THEN
* ... reset plot environment to defaults
	   CALL DISP_RESET
*          set the default scaling factor for the scene - SQRT(0.7)
           scl = dflt_imgscale
           wn_xpixels(wsid) = wn_xinches(wsid) * windowdpix(wsid) * scl
           wn_ypixels(wsid) = wn_yinches(wsid) * windowdpiy(wsid) * scl
           CALL FGD_SEND_IMAGE_SCALE(wsid, scl)
	ELSEIF ( num_args .GT. 0 ) THEN
* ... pass a single command to PLOT5
	   CALL PPLCMD (' ',' ',0,cmnd_buff(arg_start(1):len_cmnd),1,1)
	ELSE
* ... enter PPLUS interactively
	   CALL PPLCMD ( '/dev/tty', ' ', 0, ' ', 1, 0 )
	   ppl_interrupted = .FALSE.			! since interactive
	ENDIF

* check to see if mouse input was done -- take action if so (6/99)
	IF (has_new_mouse) THEN
	   jnl_out = mode_journal .AND. jrnl_lun .NE. unspecified_int4
	   ver_out = mode_verify 
* ... define symbols and output them as comments to user and journal file
	   vbuff = ' '
	   WRITE ( vbuff, '(1PG16.7)' ) xmouse
           WRITE ( ppl_buff, 3100 ) 'X', vbuff
           CALL PPLCMD ( from, line, 0, ppl_buff, 1, 1 )
	   WRITE ( ppl_buff, 3200 ) 'X', vbuff
	   slen = TM_LENSTR1(ppl_buff)
           IF (jnl_out) WRITE (jrnl_lun,'(A)') ppl_buff(:slen)
	   IF (ver_out) CALL SPLIT_LIST(pttmode_ops, ttout_lun,
     .					ppl_buff, slen )

	   vbuff = ' '
	   WRITE ( vbuff, '(1PG16.7)' ) ymouse
           WRITE ( ppl_buff, 3100 ) 'Y', vbuff
           CALL PPLCMD ( from, line, 0, ppl_buff, 1, 1 )
	   WRITE ( ppl_buff, 3200 ) 'Y', vbuff
	   slen = TM_LENSTR1(ppl_buff)
           IF (jnl_out) WRITE (jrnl_lun,'(A)') ppl_buff(:slen)
	   IF (ver_out) CALL SPLIT_LIST(pttmode_ops, ttout_lun,
     .					ppl_buff, slen )
     

	   vbuff = ' '
	   WRITE ( vbuff, '(I3)' ) wmouse
           WRITE ( ppl_buff, 3100 ) 'W', vbuff
           CALL PPLCMD ( from, line, 0, ppl_buff, 1, 1 )
	   WRITE ( ppl_buff, 3200 ) 'W', vbuff
	   slen = TM_LENSTR1(ppl_buff)
           IF (jnl_out) WRITE (jrnl_lun,'(A)') ppl_buff(:slen)
	   IF (ver_out) CALL SPLIT_LIST(pttmode_ops, ttout_lun,
     .					ppl_buff, slen )

* ... house-keeping
	   has_new_mouse = .FALSE.		! RESET !!
	   IF (jnl_out) THEN
#ifdef AIX_XLF
              CALL FLUSH(jrnl_lun)
#endif
#ifdef solaris
              CALL FLUSH(jrnl_lun)
#endif
	   ENDIF
	ENDIF

	i = STR_UPCASE( vbuff, cmnd_buff(arg_start(1):arg_end(1)))
	IF (( INDEX(vbuff,'SHADE')   .GT. 0 ) .OR.
     .      ( INDEX(vbuff,'CONTOUR') .GT. 0 ) .OR.
     .      ( INDEX(vbuff,'FILL')    .GT. 0 ) .OR.
     .      ( INDEX(vbuff,'WIRE')    .GT. 0 )    ) THEN 
	   CALL DECREMENT_PLOT_GRID  ! see ticket 883
	ENDIF

	IF (( INDEX(vbuff,'SHADE')   .GT. 0 ) .OR.
     .      ( INDEX(vbuff,'CONTOUR') .GT. 0 ) .OR.
     .      ( INDEX(vbuff,'FILL')    .GT. 0 ) .OR.
     .      ( INDEX(vbuff,'PLOT')    .GT. 0 ) .OR.
     .      ( INDEX(vbuff,'POLY')    .GT. 0 ) .OR.
     .      ( INDEX(vbuff,'WIRE')    .GT. 0 )  .OR.
     .      ( INDEX(vbuff,'RIBB')    .GT. 0 )   ) THEN 
	   IF (iaxset .EQ. 1) THEN   ! If /AXES was used on prev. plot call, reset defaults
	      CALL PPL_AXES_RESTORE
	      CALL PPLCMD ( from, line, 0, 'AXSET,1,1,1,1', 1, 1 )
	      CALL PPLCMD ( from, line, 0, 'AXLABP -1,-1', 1, 1)
	      iaxset = 0
	   ENDIF

* Also restore color palette

           IF ( set_palette ) CALL PPL_SHASET( 'SPECTRUM' )

	ENDIF

	IF (( INDEX(vbuff,'SHADE')   .GT. 0 ) .OR.
     .      ( INDEX(vbuff,'CONTOUR') .GT. 0 ) .OR.
     .      ( INDEX(vbuff,'FILL')    .GT. 0 ) .OR.
     .      ( INDEX(vbuff,'VECTOR')  .GT. 0 ) .OR.
     .      ( INDEX(vbuff,'POLY')    .GT. 0 ) .OR.
     .      ( INDEX(vbuff,'WIRE')    .GT. 0 )    ) THEN 

	   CALL RESTORE_CURVI   


	ENDIF     

* check to see if ylen or yorg has been changed
* if so, parameters saved prior to distorting yaxis are invalidated
        IF ( ylen .NE. temp_ylen ) ppl_saved_yaxlen = ylen
        IF ( yorg .NE. temp_yorg ) ppl_saved_yorg   = yorg

 1000	RETURN

 3100   FORMAT('SET ',A1,'MOUSE ', A)       ! "SET ?MOUSE position_string"
 3200	FORMAT(' !-> DEFINE SYMBOL ',A1,'MOUSE = ',A)
 
 5300   CALL ERRMSG (ferr_file_not_found, status, 
     .          'unable to open "' //
     .	fname(:slen)//'.spk" to set plot colors or pattern.', *1000)

	END
