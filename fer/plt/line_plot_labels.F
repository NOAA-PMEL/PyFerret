	SUBROUTINE LINE_PLOT_LABELS (var1, nmv, ndv, cx_list, 
     .		this_no_range, overlay, versus, nokey, time_axis, 
     .		tstyle, cal_id_1, ribbon_var, indep_lab, dep_lab, 
     .		ind_min, dep_len, hlen, nfeatures, dset, ivar)

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
* PyFr *kms*  7/13 Scale default lettering sizes by textscale
* V71  *acm* 2/17  More on ticket 2231. Label set of line plots with their units
*                  if units are different. If PLOT/KEY=title given, use the variable
*                  title in the legend labels for both PLOT and PLOT/VS plots.
* v7.2 *acm* 4/2017 ticket 2446 Drawing formatted time axes on a PLOT/VS, label
*                  at the top with the year if needed.
* V74+ *acm*  4/18 additions for plotting trajectories from a dsg dataset
* V74+ 4/18 *acm*  Automatically do a colored trajectory plot for the command "PLOT var" 
*                  if it's a DSG trajectory set.
* V74+ 7/18 *acm*  Get the title right if its a dsg data in an /overlay.
* V74+ 7/18 *acm*  DSG-enabled Ferret: colorkey labels may be strings from the IDs
* V74+ 8/18 *acm*  DSG-enabled Ferret: label the independent axis with the variable title 
*                  as well as the units, 2-line label on a short axis e.g. in a viewport
* V74+ 3/19 *acm*  If we are plotting lines from a DSG dataset that have been regridded
*                  to an ordinary time axis, then the labels are just the line labels, not
*                  feature-id labels
* V75+ 5/19 *acm*  DSG-enabled Ferret: If there is only one feature (e.g. masked) then 
*                  multi-var plots are labeled as for non-DSG data, titles of each different 
*                  variable
* V751 6/19 *acm*  When multiple datasets are open need to take care to get the right feature-id var
* V751 7/19 *acm*  DSG-enabled Ferret: PLOT/KEY for DSG data will make a line legend consistent
*                  with multi-feature plots, even if there's just one feature.
* V751 7/19 *acm*  DSG-enabled Ferret: labels when plotting user-defined variables
* V751 8/19 *acm*  DSG-enabled Ferret: labels for modifications such as regridding operation
* V76  *acm* 11/19 Issue 1960: Place keys for overlay lines in multiple columns 
*                  below plot before resorting to putting them inside the plot box.
*                  If still too many legend lines, skip them rather than put up dialog box
* V760 *acm* 3/20 Customizations for case when constraints result in no features

        IMPLICIT NONE
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
	include 'xvariables.cmn'
	include 'xdset_info.cmn_text'
	include 'xdsg_context.cmn'
        include 'netcdf.inc'

* calling argument declarations:
	LOGICAL this_no_range(*), overlay, versus, nokey, time_axis
	INTEGER var1, nmv, ndv, cx_list(*), cal_id_1, ribbon_var, 
     .		indep_lab, dep_lab, nfeatures, dset, ivar
	REAL	dep_len, hlen
	REAL*8  ind_min
	CHARACTER*(*) tstyle

* internal variable declarations:
	LOGICAL TM_ITSA_DSG, user_title, units_same, do_units, its_dsg,
     .          reset_dsg, process_feature(nfeatures), yes_key, lonlat
	INTEGER	TM_LENSTR1, TM_LENSTR, TM_UNIT_ID, STR_SAME, 
     .		TM_DSG_DSET_FROM_GRID, TM_DSG_NF2FEATURES,
     .		ipl, nline_in_mem, slen, slen1, status, ptitl, style, 
     .		justify, iunits, istart, i, nfeatr(nfeatures), dtype, 
     .		fvar_varid, dset_dsg, nlab, cx, nftrsets

	CHARACTER*2040	FULL_VAR_TITLE, VAR_TITLE_ONLY, KEY_STRING, 
     .			plot_title, plot_title_1
	CHARACTER	VAR_UNITS*64, SECS_TO_DATE*20, TM_FMT*48,
     .			SANITARY_VAR_CODE*128, VAR_TITLE_MOD*240, 
     .			PAREN_SURROUND*240,CD_DSG_FEATURENAME*16,
     .			year*4, buff1*16, ustring*64, dummy*1024, buff2*40
	REAL		xkeypos, ykeypos, xkeysize, ykeysize, 
     .			lab_ht, labsize, value
	REAL*4		SYMWID, rht
	
        LOGICAL, DIMENSION(:), ALLOCATABLE :: process_ftrset

	ptitl = qual_given( slash_title )
	user_title = ptitl .GT. 0

	its_dsg = .FALSE.
	reset_dsg = .FALSE.
	lonlat = .FALSE.
	
	IF (cxdsg_empty_set) THEN
	   buff1 = CD_DSG_FEATURENAME ( cxdsg_orient )
	   buff2 = 'No '//buff1(:TM_LENSTR1(buff1))//
     .		       ' Data in Selection'
	ENDIF

* * * * * * *    - - - TITLES - - -   * * * * * * * * *
* if we are starting from a blank screen titling information is centered
* below the X axis - either a title for a single variable or keys for multiple
* if we are overlaying on an existing plot put the new variable name on as a
* key at the lower left

	nftrsets = 0
	IF (nfeatures.GT.0) THEN
	   its_dsg = .TRUE.
	   cx = is_cx(var1)

	   DO i = 1, nfeatures
	      nfeatr(i) = -1
	   ENDDO

* create a mask showing which features were plotted, get a list of feature-numbers to label
* This may be a user-defined variable or expression on a DSG grid. Get the DSG dataset number.

	   dset_dsg = TM_DSG_DSET_FROM_GRID( cx_grid(cx) )
	   IF (dset_dsg .LT. pdset_irrelevant) THEN 

* a trajectoryProfile or timeseriesProfile dataset where we've changed the
* contexts to plot as trajectory or timeseries. Get back the context for the
* overall plot and set up the masks

	      dset_dsg = dset
	      nftrsets = TM_DSG_NF2FEATURES (dset_dsg)
	      cx = cx_buff
	      CALL MAKE_DSG_FEATURE_MASK(dset_dsg, cx_list(var1), 
     .                                   process_feature, nfeatures)
	         
	      ALLOCATE (process_ftrset(nftrsets )) 
	      CALL MAKE_DSG_FTRSET_MASK(dset, cx, nfeatures, process_feature, 
     .           nftrsets, process_ftrset)
     
	      ipl = 0
	      DO i = 1, nftrsets
	         IF (process_ftrset(i)) THEN
	            ipl = ipl + 1
		    nfeatr(ipl) = i
	         ENDIF
	      ENDDO

* The varid of the ftrset-id variable (trajectory or station)
 
	      dummy = SANITARY_VAR_CODE(cat_file_var, dsg_ftrset_var(dset_dsg) )
	      CALL CD_GET_VAR_ID  (dset_dsg, dummy, fvar_varid, status)

	   ELSE  ! feature-mask

	      CALL MAKE_DSG_FEATURE_MASK(dset_dsg, cx_list(var1), 
     .                                   process_feature, nfeatures)
	      ipl = 0
	      DO i = 1, nfeatures
	         IF (process_feature(i)) THEN
	            ipl = ipl + 1
		    nfeatr(ipl) = i
	         ENDIF
	      ENDDO

* The varid of the feature-id variable
 
	      dummy = SANITARY_VAR_CODE(cat_file_var, dsg_feature_var(dset_dsg) )
	      CALL CD_GET_VAR_ID  (dset_dsg, dummy, fvar_varid, status)

	   ENDIF


	ENDIF

* One feature but multi-vars. Label with variable names.  Add the feature-id
* to the dataset title above the plot.

	IF (its_dsg .AND. (ipl.EQ.1 .AND. (nmv-var1)+1.GT.1) ) THEN
	   reset_dsg = .TRUE.
	   its_dsg = .FALSE.  ! multi-var, one feature
	   
	   CALL CD_READ_ONE ( dset_dsg, fvar_varid, nfeatr(ipl), dtype, value, 
     .                        plot_title_1, slen, status )
	   CALL GET_DSET_TITLE (dset_dsg, dummy, slen)
	   slen = TM_LENSTR(dummy)
	   IF (slen .GT. 0) THEN
	      CALL SET_DSET_TITLE (dset_dsg, dummy(:TM_LENSTR1(dummy))//': ID='//plot_title_1)
	   ELSE
	      CALL SET_DSET_TITLE (dset_dsg, 'ID: '//plot_title_1)
	   ENDIF
	ENDIF

  400	CONTINUE
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
	         IF (.NOT.use_keys) plot_title = VAR_TITLE_ONLY( cx_list(ipl), slen )
	      ENDIF
              IF (this_no_range(ipl) .AND. mode_nodata_lab) THEN
                 plot_title(slen+1:slen+14) = ' No Valid Data'
                 slen = slen + 14
              ENDIF
              IF (.NOT. nokey) THEN
	         CALL KEY_LOC( overlay, xkeypos, ykeypos, xkeysize, ykeysize,
     .			    lab_ht, style, justify, plot_title(:slen))
* key_loc returns xkeypos=-999 if it has run out of space.  Skip this key entry.
* Could notify the user ...
		 IF (INT(xkeypos) .EQ. unspecified_int4) THEN
c		    CALL WARN ('Too many lines for legend. Not labeling "'//
c     .			plot_title(:slen)//'")
		 ELSE
		    CALL PUT_KEY( nline_in_mem, plot_title(:slen),
     .			    xkeypos, ykeypos, xkeysize, ykeysize,
     .			    lab_ht, style, lab_loc_absolute, lab_loc_absolute )
		 ENDIF
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
           ELSE
              IF ( tstyle.EQ.'MON' .OR. tstyle.EQ.'DAY' ) THEN  ! plot/vs with time variable
	         i = ind_min
                 WRITE (year, '(I4)') i
              ENDIF 
	   ENDIF

* *ACM* if time axis, then cal_id_1 is set to the calendar.  Label the
*       plot with the calendar name if there is non-standard calendar

	   CALL DISP_LABELS( cx_plot, year, cal_id_1 )

* If there are multiple variables with different units, do not label 
* the vertical axis with the units 

	   units_same = .FALSE.
	   ustring = VAR_UNITS(cx_list(var1))
	   IF (nmv .GT. var1) units_same = .TRUE.
	   DO 300 ipl = var1+1, nmv
	      IF (STR_SAME(ustring, VAR_UNITS(cx_list(ipl)) ) .NE. 0) 
     .	          units_same = .FALSE.
 300	   CONTINUE
	   IF (nmv.EQ.var1 .AND. .NOT.versus) units_same = .TRUE.

* ... label dependent axis with units if all variables have the same units
*     (over-written below if /vs)

	   plot_title = VAR_UNITS( cx_list(var1) )
	   slen = TM_LENSTR1( plot_title )
	   IF (units_same) CALL BOX_LABEL( dep_lab,
     .			   plot_title(:slen),
     .                    0.0, 0.0, 0.6*dep_len, dflt_letsize_label*textscale,
     .                    dflt_letsize_label*textscale, ppl_centered,

     .			   lab_loc_absolute, lab_loc_absolute )

* ... keys for the dependent vars, or if one, only the title or axis label
*     To do a line-legend key even for one line, issue PLOT command with PLOT/KEY

	   DO 600 ipl = var1, nmv
	      nline_in_mem = nline_in_mem + 1

	      yes_key = use_keys .AND. (.NOT. nokey) .AND.nmv.GT.var1
	      IF (its_dsg .AND. nmv.EQ.1) yes_key = qual_given(slash_plot_key) .GT. 0

	      IF ( yes_key) THEN

	         IF ( its_dsg ) THEN 
		    IF (nfeatr(ipl) .LT. 0) CYCLE

* Read the feature-level value. If it's numeric turn it into a string

                    CALL CD_READ_ONE ( dset_dsg, fvar_varid, nfeatr(ipl), dtype, value, 
     .                                 plot_title, slen, status )
		    IF (dtype .NE. NF_CHAR) plot_title = TM_FMT(value, 15, 20, slen)

	         ELSE 
	            plot_title = KEY_STRING( cx_list(ipl), key_doc, slen )
	         ENDIF 

	      ELSE
	         IF (units_same .AND. .NOT.use_keys) THEN 
	            plot_title = VAR_TITLE_ONLY( cx_list(ipl), slen )
	         ELSE
		    do_units = .TRUE.
		    IF (nmv.EQ.var1 .AND. .NOT.versus) do_units = .FALSE.
		    plot_title = FULL_VAR_TITLE( cx_list(ipl), do_units, slen )
		 ENDIF
	      ENDIF
              IF (this_no_range(ipl) .AND. mode_nodata_lab) THEN
                 plot_title(slen+1:slen+14) = ' No Valid Data'
                 slen = slen + 14
              ENDIF

	      IF ( .NOT.yes_key .AND. (ndv .EQ. 1 .OR. user_title) ) THEN
	         IF ( versus ) THEN     ! ("xlen" changed to "dep_len" 11/92)
	            CALL BOX_LABEL( dep_lab,
     .			   plot_title(:slen),
     .			   0.0, 0.0, 0.6*dep_len, dflt_letsize_label*textscale,
     .			   dflt_letsize_label*textscale, ppl_centered,
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
     .			   0.0, 0.0, 0.75*hlen, dflt_letsize_title*textscale,
     .			   dflt_letsize_title*textscale, ppl_centered,
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
	      do_units = .TRUE.
	      plot_title = FULL_VAR_TITLE( cx_list(1), do_units, slen )
              slen = TM_LENSTR1( plot_title )
              CALL BOX_LABEL( indep_lab,
     .                     plot_title(:slen),
     .                     0.0, 0.0, 0.6*hlen, dflt_letsize_label*textscale,
     .                     dflt_letsize_label*textscale, ppl_centered,
     .                     lab_loc_absolute, lab_loc_absolute )
           ELSE
	      IF (its_dsg) THEN

* Key was labeled with the feature-ids. Need the variable title, so put it on indep axis
	         IF (yes_key) THEN
		    plot_title = VAR_TITLE_ONLY( cx_list(var1), slen )

		    IF ( user_title ) CALL EQUAL_STR_LC(
     .			cmnd_buff(qual_start(ptitl):qual_end(ptitl)),
     .                  plot_title, status )

		    slen = TM_LENSTR1( plot_title )
		    plot_title_1 = plot_title(:slen)//
     .                     ' ('//ustring(:TM_LENSTR1(ustring))//')'

		    slen = TM_LENSTR1( plot_title_1 )
		    rht = dflt_letsize_label
		    IF (ylen .LT. SYMWID(rht, slen, plot_title_1) ) THEN
		       plot_title_1 = plot_title(:TM_LENSTR1(plot_title))//
     .                     '<NL>('//ustring(:TM_LENSTR1(ustring))//')'
		    ENDIF

		    slen = TM_LENSTR1( plot_title_1 )

		    plot_title = VAR_TITLE_MOD(cx_list(var1))
		    plot_title_1(slen+2:) = PAREN_SURROUND(plot_title,slen1)
		    slen = TM_LENSTR1( plot_title_1 )

		    CALL BOX_LABEL( dep_lab,
     .			   plot_title_1(:slen),
     .			   0.0, 0.0, 0.8*dep_len, dflt_letsize_label,
     .			   dflt_letsize_label, ppl_centered,
     .			   lab_loc_absolute, lab_loc_absolute )
	         ELSE  ! no keys, label w/ var title below the plot
		    plot_title = FULL_VAR_TITLE( cx_list(var1), do_units, slen )
		    IF ( user_title ) CALL EQUAL_STR_LC(
     .			cmnd_buff(qual_start(ptitl):qual_end(ptitl)),
     .                  plot_title, status )
		    slen = TM_LENSTR1( plot_title )
	            CALL BOX_LABEL( ppl_title,
     .			   plot_title(:slen),
     .			   0.0, 0.0, 0.75*hlen, dflt_letsize_title,
     .			   dflt_letsize_title, ppl_centered,
     .			   lab_loc_absolute, lab_loc_absolute )
		 ENDIF
	      ENDIF
	     
           ENDIF
	ENDIF  ! overlay

	IF (cxdsg_empty_set) THEN
	   CALL DSG_LABEL_NOFEATURE ( buff2, dflt_letsize_title, status)
	   IF (status .NE. ferr_ok) GOTO 5000 
	ENDIF

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

	         IF ( .NOT. its_dsg ) plot_title = KEY_STRING( cx_list(ipl), cx_xtra_doc, slen )

		 IF (.NOT.use_keys .OR. its_dsg) plot_title = VAR_TITLE_ONLY( cx_list(ipl), slen )
	      ENDIF
              IF (this_no_range(ipl) .AND. mode_nodata_lab) THEN
                 plot_title(slen+1:slen+14) = ' No Valid Data'
                 slen = slen + 14
              ENDIF

              IF (ipl.EQ.ribbon_var .AND. .NOT.its_dsg) THEN
		 slen1 = TM_LENSTR1( plot_title_1 )
                 plot_title = plot_title_1(1:slen1)//
     .			', Colored by '//plot_title(1:slen)
	         slen = TM_LENSTR1( plot_title)
              ENDIF

              plot_title_1 = plot_title ! ribbon plot: combine with color-by label
 1500	   CONTINUE
           IF (.NOT. nokey) THEN
              CALL KEY_LOC( overlay, xkeypos, ykeypos, xkeysize, ykeysize,
     .			    lab_ht, style, justify, plot_title(:slen))
              CALL PUT_KEY( nline_in_mem, plot_title(:slen),
     .			    xkeypos, ykeypos, xkeysize, ykeysize,
     .			    lab_ht, style, lab_loc_absolute, lab_loc_absolute )
           ENDIF

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
           ELSE
              IF ( tstyle.EQ.'MON' .OR. tstyle.EQ.'DAY' ) THEN  ! plot/vs with time variable
	         i = ind_min
                 WRITE (year, '(I4)') i
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
	   IF (units_same) CALL BOX_LABEL( dep_lab,
     .			   plot_title(:slen),
     .			   0.0, 0.0, 0.6*dep_len, dflt_letsize_label*textscale,
     .			   dflt_letsize_label*textscale, ppl_centered,
     .			   lab_loc_absolute, lab_loc_absolute )
	   ENDIF

	   IF (versus .AND. ribbon_plot) THEN
	      do_units = .TRUE.
	      plot_title = FULL_VAR_TITLE( cx_list(var1), do_units, slen )
	      slen = TM_LENSTR1( plot_title )
	      CALL BOX_LABEL( dep_lab,
     .			   plot_title(:slen),
     .			   0.0, 0.0, 0.6*dep_len, dflt_letsize_label*textscale,
     .			   dflt_letsize_label*textscale, ppl_centered,
     .			   lab_loc_absolute, lab_loc_absolute )
     
	      IF (nmv .EQ. var1) THEN
	         plot_title = 'Longitude (degrees_east)'
	         slen = TM_LENSTR1( plot_title )
	         CALL BOX_LABEL( indep_lab,
     .                     plot_title(:slen),
     .                     0.0, 0.0, 0.6*hlen, dflt_letsize_label,
     .                     dflt_letsize_label, ppl_centered,
     .                     lab_loc_absolute, lab_loc_absolute )

	         plot_title = 'Latitude (degrees_north)'
	         slen = TM_LENSTR1( plot_title )	         
	         CALL BOX_LABEL( dep_lab,
     .			   plot_title(:slen),
     .			   0.0, 0.0, 0.6*dep_len, dflt_letsize_label,
     .			   dflt_letsize_label, ppl_centered,
     .			   lab_loc_absolute, lab_loc_absolute )
	         lonlat = .TRUE.
	      ENDIF
	   ENDIF


	   IF (cxdsg_empty_set) THEN
              CALL DSG_LABEL_NOFEATURE ( buff2, dflt_letsize_title, status)
	      IF (status .NE. ferr_ok) GOTO 5000 
	   ENDIF

* ... keys for the dependent vars, or if one, only then title or axis label
! ribbon_plot

	   IF (nmv.EQ.1 .AND. (versus.AND.ribbon_plot)) THEN
	      IF (cxdsg_empty_set) do_units = .FALSE.
	      plot_title = FULL_VAR_TITLE( cx_list(nmv), do_units, slen )
              IF (this_no_range(nmv) .AND. mode_nodata_lab) THEN
                 plot_title(slen+1:slen+14) = ' No Valid Data'
                 slen = slen + 14
              ENDIF
	      
              labsize = dflt_letsize_title
              IF (.NOT.cxdsg_empty_set) plot_title = 
     .                'Colored by '//plot_title(1:slen)

              IF ( user_title ) THEN     ! user over-ride of title
                 CALL EQUAL_STR_LC
     .             ( cmnd_buff(qual_start(ptitl):qual_end(ptitl)),
     .               plot_title, status )
                 IF ( status .NE. ferr_ok ) GOTO 5000
                 labsize = dflt_letsize_title
              ENDIF

              slen = TM_LENSTR1( plot_title )
              CALL BOX_LABEL( ppl_title,
     .			   plot_title(:slen),
     .			   0.0, 0.0, 0.75*hlen, dflt_letsize_title,
     .			   labsize, ppl_centered,
     .			   lab_loc_absolute, lab_loc_absolute )
              GOTO 5000
	   ENDIF 

	   istart = var1
	   IF (.NOT.lonlat .AND. versus .AND. ribbon_plot) istart = 1

	   DO 1600 ipl = istart, var1+1
	      IF (ipl.EQ.var1 .AND. (versus.AND.ribbon_plot)) GOTO 1600  
	      nline_in_mem = nline_in_mem + 1
	      do_units = .TRUE.
	      plot_title = FULL_VAR_TITLE( cx_list(ipl), do_units, slen )
              IF (this_no_range(ipl) .AND. mode_nodata_lab) THEN
                 plot_title(slen+1:slen+14) = ' No Valid Data'
                 slen = slen + 14
              ENDIF
	         
              labsize = dflt_letsize_title*textscale
              IF (ipl.EQ.ribbon_var) THEN
	         IF (versus) THEN
	            IF (add_ribbon) plot_title_1 = 'Trajectory'
	            IF (add_ribbon) plot_title = 'Trajectory'
                    slen1 = TM_LENSTR1( plot_title_1 )
		    CALL BOX_LABEL( indep_lab,
     .                     plot_title_1(:slen1),
     .                     0.0, 0.0, 0.75*hlen, dflt_letsize_label*textscale,
     .                     dflt_letsize_label*textscale, ppl_centered,
     .                     lab_loc_absolute, lab_loc_absolute )
		    plot_title = 'Colored by '//plot_title(1:slen)
                 ELSE
		    slen1 = TM_LENSTR1( plot_title_1 )
                    plot_title = plot_title_1(1:slen1)//
     .			'<NL>Colored by '//plot_title(1:slen)
                 ENDIF

                 IF ( user_title ) THEN     ! user over-ride of title
                    CALL EQUAL_STR_LC
     .			 ( cmnd_buff(qual_start(ptitl):qual_end(ptitl)),
     .                    plot_title, status )
                    IF ( status .NE. ferr_ok ) GOTO 5000
                    labsize = dflt_letsize_title*textscale
	         ENDIF
              ENDIF

              slen = TM_LENSTR1( plot_title )
              CALL BOX_LABEL( ppl_title,
     .			   plot_title(:slen),
     .			   0.0, 0.0, 0.75*hlen, dflt_letsize_title*textscale,
     .			   labsize, ppl_centered,
     .			   lab_loc_absolute, lab_loc_absolute )

              plot_title_1 = plot_title ! ribbon plot: combine with color-by label
 1600	   CONTINUE

	ENDIF ! overlay

5000	CONTINUE

! Restore dataset title
	IF (reset_dsg) CALL SET_DSET_TITLE (dset_dsg, dummy)

	IF (nftrsets.GT.0) DEALLOCATE (process_ftrset)

	RETURN
	END

