	SUBROUTINE LINE_PLOT_LABELS (var1, nmv, ndv, cx_list, 
     .		this_no_range, overlay, versus, nokey, time_axis, 
     .		tstyle, cal_id_1, ribbon_var, indep_lab, dep_lab, 
     .		ind_min, dep_len, hlen)

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
* Put titles on line plots. For multi-line or ribbon plots, add labels
* for multi-lines as a key, or a "colored by var2" label for the ribbon color variable
*
* Pulled out of plot_set_up.F 
* 8/12 ACM 
* V6.83 *acm* 11/30/2012 Need AXIS.INC for the axis lengths, xlen, ylen

	include 'tmap_dims.parm'
	include 'ferret.parm'
	include 'plot_setup.parm'
	include 'errmsg.parm'
	include 'slash.parm'
	include 'xplot_setup.cmn'
	include 'xprog_state.cmn'
	include 'xplot_state.cmn'
	include 'axis_inc.decl'    ! with axis lengths 
	include 'AXIS.INC'         ! with axis lengths   
	include 'xcontext.cmn'

* calling argument declarations:
	LOGICAL this_no_range(*), overlay, versus, nokey, time_axis
	INTEGER var1, nmv, ndv, cx_list(*), cal_id_1, ribbon_var, 
     .		indep_lab, dep_lab
	REAL	dep_len, hlen
	REAL*8  ind_min
	CHARACTER*(*) tstyle

* internal variable declarations:
	LOGICAL user_title
	INTEGER	TM_LENSTR1, TM_UNIT_ID, ipl, nline_in_mem, slen, slen1, 
     .		status, ptitl, style, justify, iunits, istart
	
	CHARACTER*2040	FULL_VAR_TITLE,  KEY_STRING, plot_title, plot_title_1
	CHARACTER	VAR_UNITS*64, SECS_TO_DATE*20, year*4, buff1*16
	REAL		xkeypos, ykeypos, xkeysize, ykeysize, 
     .			lab_ht, labsize

	ptitl = qual_given( slash_title )
	user_title = ptitl .GT. 0

* * * * * * *    - - - TITLES - - -   * * * * * * * * *
* if we are starting from a blank screen titling information is centered
* below the X axis - either a title for a single variable or keys for multiple
* if we are overlaying on an existing plot put the new variable name on as a
* key at the lower left

	IF (ribbon_plot) GOTO 1000


* * * * * * *    - - - TITLES - - -   * * * * * * * * *
* if we are starting from a blank screen titling information is centered
* below the X axis - either a title for a single variable or keys for multiple
* if we are overlaying on an existing plot put the new variable name on as a
* key at the lower left
	nline_in_mem = 0
	IF ( overlay ) THEN
	   DO 500 ipl = var1, nmv
	      nline_in_mem = nline_in_mem + 1
	      CALL OV_GRP_CX_DOC( cx_list(ipl), 1, cx_xtra_doc )
	      IF ( user_title ) THEN
	         CALL EQUAL_STR_LC(
     .			cmnd_buff(qual_start(ptitl):qual_end(ptitl)),
     .                  plot_title, status )
                 IF ( status .NE. ferr_ok ) GOTO 5000
	         slen = TM_LENSTR1( plot_title )
	      ELSE
	         plot_title = KEY_STRING( cx_list(ipl), cx_xtra_doc, slen )
	      ENDIF
              IF (this_no_range(ipl) .AND. mode_nodata_lab) THEN
                 plot_title(slen+1:slen+14) = ' No Valid Data'
                 slen = slen + 14
              ENDIF
              IF (.NOT. nokey) THEN
	         CALL KEY_LOC( overlay, xkeypos, ykeypos, xkeysize, ykeysize,
     .			    lab_ht, style, justify, plot_title(:slen))
	         CALL PUT_KEY( nline_in_mem, plot_title(:slen),
     .			    xkeypos, ykeypos, xkeysize, ykeysize,
     .			    lab_ht, style, lab_loc_absolute, lab_loc_absolute )
              ENDIF
 500	   CONTINUE
	ELSE
* *kob* 1/00 check the variable time_axis before looking at
*            any time-relevant variables.
* ... basic position and data set labels 
*    ... does the year need special documentation? 
           year = unspecified_name4    ! assume no label needed by default 
           IF ( time_axis ) THEN 
*    ... document the year where the axis, itself, doesn't 
              IF ( tstyle.EQ.'MON' .OR. tstyle.EQ.'DAY' ) THEN 
                 ppl_buff = SECS_TO_DATE( ind_min, cal_id_1 ) 
                 year = ppl_buff(8:11) 
              ENDIF 
           ENDIF 
* *ACM* if time axis, then cal_id_1 is set to the calendar.  Label the
*       plot with the calendar name if there is non-standard calendar
	   CALL DISP_LABELS( cx_plot, year, cal_id_1 )
* ... label dependent axis with units (over-written below if /vs)
	   plot_title = VAR_UNITS( cx_list(var1) )
	   slen = TM_LENSTR1( plot_title )
	   CALL BOX_LABEL( dep_lab,
     .			   plot_title(:slen),
     .			   0.0, 0.0, 0.6*dep_len, letsize_label*pplscale,
     .			   letsize_label*pplscale, ppl_centered,
     .			   lab_loc_absolute, lab_loc_absolute )
* ... keys for the dependent vars, or if one, only then title or axis label
	   DO 600 ipl = var1, nmv
	      nline_in_mem = nline_in_mem + 1
	      IF ( use_keys .AND. (.NOT. nokey) ) THEN
	         plot_title = KEY_STRING( cx_list(ipl), key_doc, slen )
	      ELSE
	         plot_title = FULL_VAR_TITLE( cx_list(ipl), slen )
	      ENDIF
              IF (this_no_range(ipl) .AND. mode_nodata_lab) THEN
                 plot_title(slen+1:slen+14) = ' No Valid Data'
                 slen = slen + 14
              ENDIF
	      IF ( ndv .EQ. 1 .OR. user_title ) THEN
	         IF ( versus ) THEN     ! ("xlen" changed to "dep_len" 11/92)
	            CALL BOX_LABEL( dep_lab,
     .			   plot_title(:slen),
     .			   0.0, 0.0, 0.6*dep_len, letsize_label*pplscale,
     .			   letsize_label*pplscale, ppl_centered,
     .			   lab_loc_absolute, lab_loc_absolute )
	         ENDIF
	         IF ( .NOT.versus .OR. user_title ) THEN 
	            IF ( user_title ) THEN     ! user over-ride of title
	               CALL EQUAL_STR_LC
     .			( cmnd_buff(qual_start(ptitl):qual_end(ptitl)),
     .                    plot_title, status )
                       IF ( status .NE. ferr_ok ) GOTO 5000
	               slen = TM_LENSTR1( plot_title )
	            ENDIF
	            CALL BOX_LABEL( ppl_title,
     .			   plot_title(:slen),
     .			   0.0, 0.0, 0.75*hlen, letsize_title*pplscale,
     .			   letsize_title*pplscale, ppl_centered,
     .			   lab_loc_absolute, lab_loc_absolute )
	         ENDIF
	      ELSE
                 IF (.NOT. nokey) THEN
	            CALL KEY_LOC( overlay, xkeypos, ykeypos, xkeysize, ykeysize,
     .			       lab_ht, style, justify, plot_title(:slen) )
	            CALL PUT_KEY( nline_in_mem, plot_title(:slen),
     .			       xkeypos, ykeypos, xkeysize, ykeysize,
     .			       lab_ht, style,
     .			       lab_loc_absolute, lab_loc_absolute )
	         ENDIF
	      ENDIF

 600	   CONTINUE

           IF (versus) THEN
	      plot_title = FULL_VAR_TITLE( cx_list(1), slen )
              slen = TM_LENSTR1( plot_title )
              CALL BOX_LABEL( indep_lab,
     .                     plot_title(:slen),
     .                     0.0, 0.0, 0.6*hlen, letsize_label*pplscale,
     .                     letsize_label*pplscale, ppl_centered,
     .                     lab_loc_absolute, lab_loc_absolute )
           ENDIF


	ENDIF  ! overlay
	
	GOTO 5000  ! done


* Labels for ribbon plots.
 1000	CONTINUE

	nline_in_mem = 0
	IF ( overlay ) THEN
	   DO 1500 ipl = var1, nmv
	      nline_in_mem = nline_in_mem + 1
	      CALL OV_GRP_CX_DOC( cx_list(ipl), 1, cx_xtra_doc )
	      IF ( user_title ) THEN
	         CALL EQUAL_STR_LC(
     .			cmnd_buff(qual_start(ptitl):qual_end(ptitl)),
     .                  plot_title, status )
                 IF ( status .NE. ferr_ok ) GOTO 5000
	         slen = TM_LENSTR1( plot_title )
	      ELSE
	         plot_title = KEY_STRING( cx_list(ipl), cx_xtra_doc, slen )
	      ENDIF
              IF (this_no_range(ipl) .AND. mode_nodata_lab) THEN
                 plot_title(slen+1:slen+14) = ' No Valid Data'
                 slen = slen + 14
              ENDIF
              IF (.NOT. nokey) THEN
	         CALL KEY_LOC( overlay, xkeypos, ykeypos, xkeysize, ykeysize,
     .			    lab_ht, style, justify, plot_title(:slen))
	         CALL PUT_KEY( nline_in_mem, plot_title(:slen),
     .			    xkeypos, ykeypos, xkeysize, ykeysize,
     .			    lab_ht, style, lab_loc_absolute, lab_loc_absolute )
              ENDIF
 1500	   CONTINUE

	ELSE ! Not overlay

* *kob* 1/00 check the variable time_axis before looking at
*            any time-relevant variables.
* ... basic position and data set labels 
*    ... does the year need special documentation? 

	   year = unspecified_name4    ! assume no label needed by default 
	   IF ( time_axis ) THEN 
*    ... document the year where the axis, itself, doesn't 
	      IF ( tstyle.EQ.'MON' .OR. tstyle.EQ.'DAY' ) THEN 
                 ppl_buff = SECS_TO_DATE( ind_min, cal_id_1 ) 
                 year = ppl_buff(8:11) 
	      ENDIF 
	   ENDIF 

* *ACM* if time axis, then cal_id_1 is set to the calendar.  Label the
*       plot with the calendar name if there is non-standard calendar

	   CALL DISP_LABELS( cx_plot, year, cal_id_1 )

* ... label dependent axis with units (over-written below if /vs)

	   plot_title = VAR_UNITS( cx_list(var1) )

	   slen = TM_LENSTR1( plot_title )
	   buff1 = VAR_UNITS(cx_list(2))  ! ? was cx_y
	   iunits = TM_UNIT_ID(buff1)
	   IF (iunits .NE. 4) THEN
	      CALL BOX_LABEL( dep_lab,
     .			   plot_title(:slen),
     .			   0.0, 0.0, 0.6*dep_len, letsize_label*pplscale,
     .			   letsize_label*pplscale, ppl_centered,
     .			   lab_loc_absolute, lab_loc_absolute )
	   ENDIF

	   IF (versus .AND. ribbon_plot) THEN
	      plot_title = FULL_VAR_TITLE( cx_list(var1), slen )
	      slen = TM_LENSTR1( plot_title )
	      CALL BOX_LABEL( dep_lab,
     .			   plot_title(:slen),
     .			   0.0, 0.0, 0.6*dep_len, letsize_label*pplscale,
     .			   letsize_label*pplscale, ppl_centered,
     .			   lab_loc_absolute, lab_loc_absolute )
	   ENDIF

* ... keys for the dependent vars, or if one, only then title or axis label
! ribbon_plot

	   istart = var1
	   IF (versus .AND. ribbon_plot) istart = 1

	   DO 1600 ipl = istart, var1+1
	      IF (ipl .EQ. var1 .AND. (versus .AND. ribbon_plot)) GOTO 1600     
	      nline_in_mem = nline_in_mem + 1
	      IF ( use_keys .AND. (.NOT. nokey) ) THEN
	         plot_title = KEY_STRING( cx_list(ipl), key_doc, slen )
	      ELSE
	         plot_title = FULL_VAR_TITLE( cx_list(ipl), slen )
	      ENDIF
              IF (this_no_range(ipl) .AND. mode_nodata_lab) THEN
                 plot_title(slen+1:slen+14) = ' No Valid Data'
                 slen = slen + 14
              ENDIF
	         
              labsize = letsize_title
              IF (ipl.EQ.ribbon_var) THEN
	         IF (versus) THEN
                    slen1 = TM_LENSTR1( plot_title_1 )
		    CALL BOX_LABEL( indep_lab,
     .                     plot_title_1(:slen1),
     .                     0.0, 0.0, 0.75*hlen, letsize_label*pplscale,
     .                     letsize_label*pplscale, ppl_centered,
     .                     lab_loc_absolute, lab_loc_absolute )
		    plot_title = 'Colored by '//plot_title(1:slen)
                 ELSE
		    slen1 = TM_LENSTR1( plot_title_1 )
                    plot_title = plot_title_1(1:slen1)//
     .			'<NL>Colored by '//plot_title(1:slen)
                 ENDIF

!		 labsize = 0.8*letsize_title

                 IF ( user_title ) THEN     ! user over-ride of title
                    CALL EQUAL_STR_LC
     .			 ( cmnd_buff(qual_start(ptitl):qual_end(ptitl)),
     .                    plot_title, status )
                    IF ( status .NE. ferr_ok ) GOTO 5000
                    labsize = letsize_title
	         ENDIF
              ENDIF

              slen = TM_LENSTR1( plot_title )
              CALL BOX_LABEL( ppl_title,
     .			   plot_title(:slen),
     .			   0.0, 0.0, 0.75*hlen, letsize_title*pplscale,
     .			   labsize*pplscale, ppl_centered,
     .			   lab_loc_absolute, lab_loc_absolute )

              plot_title_1 = plot_title ! ribbon plot: combine with color-by label
 1600	   CONTINUE

	ENDIF ! overlay


 5000	RETURN
	END
