	SUBROUTINE DISP_LABELS( joint_cx, year, cal_id )

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
* put background labels onto plot - data set name and information
* about axes not apparent on the plot (eg. time and depth in an xy view)

* programmer - steve hankin
* NOAA/PMEL, Seattle, WA - Tropical Modeling and Analysis Program
* written for VAX computer under VMS operating system
*
* revision 0.0 - 7/13/87 - extracted from DISP_SET_UP and modified
* revision 0.1 - 2/25/87 - allowed for non_geographic labels
* revision 0.2 - 4/1/88  - increased AX_TITLE size for units
* revision 0.3 - 9/8/88  - added relative position flags to PUT_LABEL calls
* revision 0.4 - 9/21/88 - don't update label positions if /NOLABEL
* 			 - added time axis style: document year if taxis doesn't
*	(bug: the year displayed should be the minimum from all the contexts.
*		... laziness prevents me. *sh* )
* revision 0.5 -  9/1/89 - suppress "N/A" labelling of normal axes
* V200:  6/27/89 - ct-->cx for stylistic consistency
*		 - base axis labelling needs on cx_lo_ww vs. mnormal
*		 - check all variables in list to see if labels are needed
*	12/19/89 - eliminated the possibility of multiple data set variables
*	  1/4/89 - major revisions incorporating keys
* Unix/RISC port 5/91 *sh*: use data set name, only (full path is too long)
* V401: 6/23/95 - tag the movable labels by passing ppl_movlab minus nnn
* V510 *sh* 3/00 - add DODS URL label
* V530 *acm* 3/01 - add calendar name (if not Gregorian calendar)
* v533 *acm* 8/01 - test for calendar label had .LT. max_calendars; needs .LE.
*      *acm* 10/01  remove VMS includes
* V580 *acm* 11/04 - bug 1098 url is cut off; keep going to the last /
* V581: *acm*/3/05 - increase length of strings for dataset title to 1024
* V68  *acm*  4/11 - more on bug 1098. Call the url the OPeNDAP label, break it up.
*       *acm* 3/12 Add E and F dimensions (use nferdims in tmap_dims.parm)
* PyFr *kms*  7/13 Moved key_xspace0, key_yspace0, key_labht0, key_xorigin, 
*                  key_yorigin, key_xwhite_space, key_ywhite_space, 
*                  key_line_frac, ovkey_space, ovkey_yspace, ovkey_labht,
*                  ovkey_xorigin, ovkey_yorigin, ul_labht0, ul_yspace0,
*                  ul_xorigin, and ul_yorigin to plot_setup.parm,
*                  since they are used as parameters, and prefixed with 
*                  dflt_ those that need to be scaled by pplscale or textscale.
* V6.95 *acm* 3/15 Changes for PyFerret fonts. 
* V7   *acm* 6/16 see ticket 2435. Tweak to split dataset labels after DATA SET:
* v74+ *acm* 5/18 See PyFerret issue #53: if the title is blank don't set up 
*                 the label or the symbol labnum_datitl
* v74+ *acm* 7/18 For plots of DSG data, write the upper-left labels as world coords
* v751 *acm* 9/18 Updates for user-defined variables on DSG grids

        IMPLICIT NONE
	include 'tmap_dims.parm'
	include 'xdset_info.cmn_text'
	external xdset_info_data
	include 'ferret.parm'
	include 'plot_setup.parm'
	include 'xplot_setup.cmn'
	include 'xplot_state.cmn'
	include 'xcontext.cmn'
	include 'xtext_info.cmn'
	include 'axis_inc.decl'	! axis lengths
	include 'AXIS.INC'	! axis lengths
	include 'calendar.decl'
	include 'calendar.cmn'

* calling argument declarations:
	INTEGER		joint_cx, cal_id
	CHARACTER	year*4

* internal variable declarations:
	LOGICAL		show_year, has_mask
	CHARACTER	AX_TITLE*32, TM_GET_CALENDAR_NAME*32,
     .			line_buff*1024, genl_buff*255, cal_buff*32
	INTEGER		TM_LENSTR, TM_DSG_DSET_FROM_GRID, 
     .			idim, len, lablen,
     .			nlabs, dset_num, tlen, lab2, grid, dset_dsg
	REAL 		ul_yspace, ul_labht, ul_ypos, ul_ysqueeze,
     .			yoff, ur_labht
	EQUIVALENCE	( ppl_buff, genl_buff )	! save space

* extra upper left label to display the year ?(in case its not on the time axis)
	nlabs = ul_nlabs
	show_year = year .NE. unspecified_name4
	IF ( show_year ) nlabs = nlabs + 1
        IF (cal_id .GT. GREGORIAN .AND. cal_id .LE. max_calendars) 
     .     nlabs = nlabs + 1

* need to reduce size of upper left labels ?
	IF     ( nlabs .LE. 2 ) THEN
	   ul_ysqueeze = 1.0
	ELSEIF ( nlabs .EQ. 3 ) THEN
	   ul_ysqueeze = 0.85
	ELSE
	   ul_ysqueeze = 0.65
	ENDIF
	ul_yspace = dflt_ul_yspace0*pplscale * ul_ysqueeze
	ul_labht  = dflt_ul_labht0*textscale * ul_ysqueeze

* define the position labels ("LATITUDE:", "DEPTH:", etc)
* ... stack them above one another in the upper left

        grid = cx_grid(joint_cx)

        dset_num = cx_data_set(joint_cx)
	dset_dsg = pdsg_not_dsg
        IF (dset_num .GT. pdset_irrelevant) THEN
	   dset_dsg = TM_DSG_DSET_FROM_GRID( grid )
           IF (dset_dsg .GT. pdset_irrelevant) THEN
	      IF ( dsg_ragged(dset_dsg) ) grid = dsg_xlate_grid(dset_dsg) 
	      cx_grid(joint_cx) = grid
	      cx_data_set(joint_cx) = dset_dsg
	      dset_dsg = TM_DSG_DSET_FROM_GRID( grid )
	   ENDIF
        ENDIF

	ul_ypos = dflt_ul_yorigin + (nlabs-1)*ul_yspace
	DO 300 idim = 1, nferdims
	   IF ( ul_dolab(idim) ) THEN
	      CALL ASCII_LIMITS( joint_cx, idim, line_buff, len )

	      IF (idim .EQ. e_dim) THEN

		 CALL GET_DSG_MASK_TITLE (dset_num, genl_buff, has_mask, tlen)
		 IF (has_mask) THEN
		    IF (tlen .GT. 0) THEN
		       line_buff = 'Subset, '//genl_buff(:tlen)
		    ELSE 
		       line_buff = '(subset selected)'
		    ENDIF
		 ELSE
		    
		 ENDIF
		 
		 len = TM_LENSTR(line_buff)

	      ENDIF

	      genl_buff = AX_TITLE( idim, grid, tlen )

	      CALL PUT_LABEL( ppl_movlab-idim, 
     .                        dflt_ul_xorigin*pplscale, ul_ypos,
     .			      ppl_left_just,
     .			      ul_labht,
     .			      genl_buff(:tlen)//' : '//line_buff(:len), 
     .			      lab_loc_absolute, lab_loc_axlen_plus )
	      ul_ypos = ul_ypos - ul_yspace
	   ENDIF
 300	CONTINUE

* ... add "YEAR : ($" ! kludge - unless year value indicates climatology
	IF ( show_year ) THEN
	   IF ( year .EQ. ' ' ) THEN
	         line_buff = 'climatology'
	         lablen = 11
	   ELSE
	         line_buff = year
	         lablen = 4
	   ENDIF
	   CALL PUT_LABEL( ppl_movlab-ppl_year, 
     .                     dflt_ul_xorigin*pplscale, ul_ypos,
     .			   ppl_left_just,
     .			   ul_labht,
     .			   'YEAR : '//line_buff(:lablen),
     .			   lab_loc_absolute, lab_loc_axlen_plus )
	   ul_ypos = ul_ypos - ul_yspace
	ENDIF

* calendar label needed ?
        IF (cal_id .GT. GREGORIAN .AND. cal_id .LE. max_calendars) THEN
           cal_buff = TM_GET_CALENDAR_NAME (cal_id)
           lablen = TM_LENSTR(cal_buff)

	   CALL PUT_LABEL( ppl_movlab-ppl_calendar, 
     .                     dflt_ul_xorigin*pplscale, ul_ypos,
     .			   ppl_left_just,
     .			   ul_labht,
     .			   'CALENDAR: '//cal_buff(:lablen), 
     .			   lab_loc_absolute, lab_loc_axlen_plus )
	ENDIF


* data set labels needed ?
	dset_num = cx_data_set( joint_cx )
	IF (  dset_num .NE. unspecified_int4
     .	.AND. dset_num .NE. pdset_irrelevant ) THEN
* ... name of data set
! 5/91	   CALL GET_DSET_NAME( dset_num, line_buff, lablen )
	   CALL GET_SHORT_DSET_NAME( dset_num, line_buff, lablen )

           yoff = dflt_ul_yorigin*pplscale
           ur_labht = dflt_ul_labht0*textscale
	   IF ( ds_des_name(dset_num)(1:7) .EQ. 'http://' ) THEN
              yoff = dflt_ul_yorigin*pplscale - 0.5*ul_yspace
              ur_labht = 0.75*dflt_ul_labht0*textscale
	   ENDIF
	   CALL BOX_LABEL( ppl_movlab-ppl_dset, 
     .			   'DATA SET: '//line_buff(:lablen),
     .			   0.0, yoff,
     .			   0.45*xlen, 1.5*dflt_letsize_label*textscale,
     .			   ur_labht, ppl_right_just,
     .			   lab_loc_axlen_plus, lab_loc_axlen_plus )

* ... title of data set
	   CALL GET_DSET_TITLE( dset_num, line_buff, lablen )
	   IF (TM_LENSTR(line_buff) .GT. 0) 
     .	   CALL BOX_LABEL( ppl_movlab-ppl_datitl, line_buff(:lablen),
     .			   0.5, 0.3, 0.8*xlen, 1.5*dflt_letsize_label,
     .			   dflt_letsize_label, ppl_centered,
     .			   lab_loc_axlen_times, lab_loc_axlen_plus )
* ... OPeNDAP URL of data set
* bug 1098 url is cut off; keep going to the last /

	   IF ( ds_des_name(dset_num)(1:7) .EQ. 'http://' ) THEN
	      lablen = INDEX(ds_des_name(dset_num)(8:),'/')
              lab2 = lablen
              DO WHILE (lab2 .GT. 0)
 	         lab2 = INDEX(ds_des_name(dset_num)(8+lablen:),'/')
                 lablen = lablen + lab2
	      ENDDO
	      lablen = INDEX(ds_des_name(dset_num)(8+lablen:),'/')
     .		     + 7 + lablen

              yoff = dflt_ul_yorigin*pplscale + 0.5*ul_yspace
	      CALL BOX_LABEL( ppl_movlab-ppl_dodsurl, 
     .			   'OPeNDAP URL: '//ds_des_name(dset_num)(:lablen),
     .			   0.0, yoff,
     .			   0.4*xlen, 1.5*dflt_letsize_label*textscale,
     .			   ur_labht, ppl_right_just,
     .			   lab_loc_axlen_plus, lab_loc_axlen_plus )

	   ENDIF
	ENDIF


	RETURN
	END
