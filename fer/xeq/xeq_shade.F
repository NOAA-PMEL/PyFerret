 	SUBROUTINE XEQ_SHADE( memory )

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

* programmer - steve hankin
* NOAA/PMEL, Seattle, WA - Tropical Modeling and Analysis Program
* written for VAX computer under VMS operating system
*
* revision 0.0 - 8/2/88 - dummy template version
* revision 0.1 - 8.29.88 J Davison
*              - Modification of XEQ_CONTOUR version 0.9
* revision 0.2 - 9/21/88 - *sh* /NOLABEL processing changed:
*		 do_labels(in DISP_SET_UP) --> no_labels(in PUT_LABEL)
* revision 0.3 - 5/1/89  - PPL LEVELS not cleared for no_range if /LEVELS_SAME
*		 - this logic, do_key and vertical_key moved before /SET_UP
* V200:  5/19/89 - 4D symmetrical
*	  1/4/89 - using keys for labelling
*	 3/13/90 - check status from DISP_PREP
* Unix/RISC port - 2/26/91 - changed an INCLUDE statement to upper case
*       10/22/92 - removed dependence on shade_vars.cmn using PPLCMND(SHAKEY)
* V230:  11/9/92 - protect colors before doing a color overlay
*                - default for KEY is off on overlays
* V300:  3/11/93 - added /SPECTRUM=
* V301:	  1/7/94 - bug fix: (see DISP_DATA_SET_UP) must unprotect plot data
* 	 1/10/94 - change to DISP_SET_UP call to accomodate CONTOUR/PEN
* V312: 5/94 - array "memory" as a calling argument
* V320:   1/20/94 - changes to SAVE_FRAME call reflecting C-based procedures
* 12/96 *kob* - Linux Port - had to move variable assignments off of
*               declaration line.
* V450:  5/29/97 - added /NOAXES qualifier (useful in curvilinear plots)

#ifdef unix
	include 'tmap_format/tmap_dims.parm'
	include	'ferret_cmn/ferret.parm'
	include 'ferret_cmn/errmsg.parm'
	include 'ferret_cmn/plot_setup.parm'
	include 'ferret_cmn/slash.parm'
	include 'ferret_cmn/xcontext.cmn'
	include 'ferret_cmn/xvariables.cmn'
	include 'ferret_cmn/xprog_state.cmn'
	include 'ferret_cmn/xplot_state.cmn'
	include 'ferret_cmn/xplot_setup.cmn'
#else
	INCLUDE 'TMAP_FORMAT:TMAP_DIMS.PARM'
	INCLUDE	'FERRET_CMN:FERRET.PARM'
	INCLUDE 'FERRET_CMN:ERRMSG.PARM'
	INCLUDE 'FERRET_CMN:PLOT_SETUP.PARM'
	INCLUDE 'FERRET_CMN:SLASH.PARM'
	INCLUDE 'FERRET_CMN:XCONTEXT.CMN'
	INCLUDE 'FERRET_CMN:XVARIABLES.CMN'
	INCLUDE 'FERRET_CMN:XPROG_STATE.CMN'
	INCLUDE 'FERRET_CMN:XPLOT_STATE.CMN'
	INCLUDE 'FERRET_CMN:XPLOT_SETUP.CMN'
#endif

* calling argument declarations:
	REAL	memory( mem_blk_size, max_mem_blks )

* internal variable declarations:
	LOGICAL	  no_range, overlay, transpz, no_labels, do_key,
     .            spectrum, set_up, noaxes
	INTEGER	  TM_LENSTR1, status, ishade, slen, sp,
     .		  ax1_blks, ax1_start,
     .		  ax2_blks, ax2_start
	REAL	  only_val, aspect
	PARAMETER ( aspect = 0.0)			!kob 12/96
	CHARACTER TM_FMT*12, fname*128

* internal parameter declarations:
	LOGICAL		norm_dims_only, not_vector
	PARAMETER     ( norm_dims_only  = .FALSE.,
     .			not_vector	= .FALSE. )

* set up the graphics environment and get the data
	CALL START_PPLUS
	CALL DISP_DATA_SET_UP( memory,
     .		ax1_start, ax1_blks, ax2_start, ax2_blks, status )
	IF ( status .NE. ferr_ok ) GOTO 5100

* initialize
	overlay   = qual_given( slash_shad_over_plot ).GT.0 .AND. twodee_on
	no_labels = qual_given( slash_shad_nolabels  ).GT.0
	transpz   = qual_given( slash_shad_transpz   ).GT.0
        sp        = qual_given( slash_cont_spectrum  )
	set_up	  = qual_given( slash_shad_set_up    ).GT.0
	noaxes    = qual_given( slash_noaxes         ).GT.0
        spectrum  = sp .GT. 0 
	IF ( .NOT.overlay ) CALL DISP_INIT(no_labels,is_cx,num_uvars_in_cmnd)
* ... default orientation for YT and ZT contours has time on horizontal
	IF ( plot_axis(2).EQ.t_dim .AND. plot_axis(1).NE.x_dim )
     .							transpz = .NOT.transpz
	CALL DISP_PREP( status )
	IF ( status .NE. ferr_ok ) GOTO 5000

* set up the axes and title and load the data into PPLUS
	CALL DISP_SET_UP   (    memory,
     .				not_vector,
     .				aspect,
     .				overlay,
     .				transpz,
     .				no_labels,
     .				no_range,
     .				only_val,
     .				is_mr,
     .				num_uvars_in_cmnd,
     .				is_cx,
     .				is_uvar,
     .				memory( 1, ax1_start ),
     .				memory( 1, ax2_start ),
     .				-1, 			! auto pen select
     .				status )
	   IF ( status .NE. ferr_ok ) GOTO 5000

* PPLUS blows up setting auto contour levels if all data have same value
	   IF ( no_range .AND. qual_given( slash_shad_lev_same ) .EQ. 0 )
     .		CALL PPLCMD ( from, line, 0, 
     .		'LEV () ('//TM_FMT(only_val, 3, 12, slen)//')' , 1, 1 )

* default key
	   do_key =  (qual_given(slash_shad_key)  .GT. 0)
     .		.OR. ( (qual_given(slash_shad_line) .EQ. 0)
     .           .AND. .NOT.overlay )
	   IF (qual_given(slash_shad_nokey) .gt. 0) do_key = .FALSE.
           IF ( do_key ) THEN
              CALL PPLCMD ( from, line, 0, 'SHAKEY 1,1', 1, 1 )
           ELSE
              CALL PPLCMD ( from, line, 0, 'SHAKEY 0,1', 1, 1 )
           ENDIF

* protect colors already on-screen before a color overlay
           IF ( overlay ) CALL PPL_SHASET( 'PROTECT' )

* save old and reset to new color spectrum if requested
           IF ( spectrum ) THEN
              CALL EQUAL_STR_LC( cmnd_buff(qual_start(sp):qual_end(sp)),
     .             fname, status )
              IF ( status .NE. ferr_ok ) RETURN
              IF ( fname .EQ. ' ' ) fname = 'default'
              CALL PPL_SHASET( 'SAVE' )
              CALL PPL_SHASET( 'SPECTRUM='//fname(1:TM_LENSTR1(fname)) )
           ENDIF

* turn off axes? (5/29/97)
	IF ( noaxes ) CALL PPL_AXES_OFF(set_up)

* /set_up only requested ?
	   IF ( set_up ) GOTO 1000

* shade it 
	   IF ( overlay ) THEN
	      if (qual_given(slash_shad_line) .gt. 0) then
		 CALL PPLCMD ( ' ', ' ', 0,
     .			'SHADE/OVER/' //ppl_wait//' ' , 1, 1 )
	         IF ( ppl_interrupted ) GOTO 1000
		 CALL PPLCMD ( ' ', ' ', 0,
     .			'CONTOUR/OVER/' //ppl_wait//' ' , 1, 1 )
	      else
	         CALL PPLCMD ( ' ', ' ', 0,
     .			'SHADE/OVER/' //ppl_wait//' ' , 1, 1 )
	      end if
	   ELSE
	      if (qual_given(slash_shad_line) .gt. 0) then
		 CALL PPLCMD ( ' ', ' ', 0,
     .			'SHADE/LATCH/'//ppl_wait//' ', 1, 1 )
	         IF ( ppl_interrupted ) GOTO 1000
		 CALL PPLCMD ( ' ', ' ', 0,
     .			'CONTOUR/OVER/' //ppl_wait//' ' , 1, 1 )
	      else
		 CALL PPLCMD ( ' ', ' ', 0,
     .			'SHADE/LATCH/'//ppl_wait//' ', 1, 1 )
	      end if
	   ENDIF

* protect the new colors and restore the original color spectrum
           IF ( spectrum ) THEN
              CALL PPL_SHASET( 'PROTECT' )
              CALL PPL_SHASET( 'SPECTRUM' )
           ENDIF

* restore axes? (5/29/97)
	IF ( noaxes ) CALL PPL_AXES_RESTORE

* interrupt occurred during plot ?
	IF ( ppl_interrupted ) GOTO 1000

* capture the bit map ? Well, OK...
	   IF ( qual_given( slash_shad_frame ) .GT. 0 ) THEN
	      CALL SAVE_FRAME( frame_file, "HDF", status )
	      IF ( status .NE. ferr_ok ) GOTO 5000
	   ENDIF

* release the memory used for axis buffers
 1000 	CALL FREE_MEMORY( ax2_start, ax2_blks )
 	CALL FREE_MEMORY( ax1_start, ax2_blks )
	CALL UNPROTECT_CMND_DATA			! 1/94
 5100	RETURN

* error exits
 5000	GOTO 1000
	END
