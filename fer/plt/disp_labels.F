	SUBROUTINE DISP_LABELS( joint_cx, year )

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

#ifdef unix
	include 'tmap_dims.parm'
	include 'xdset_info.cmn_text'
	external xdset_info_data
	include 'ferret.parm'
	include 'plot_setup.parm'
	include 'xplot_setup.cmn'
	include 'xcontext.cmn'
	include 'xtext_info.cmn'
	include 'axis_inc.decl'	! axis lengths
	include 'AXIS.INC'	! axis lengths
#else
	INCLUDE 'TMAP_FORMAT:TMAP_DIMS.PARM'
	INCLUDE 'TMAP_FORMAT:XDSET_INFO.CMN'
	INCLUDE 'FERRET_CMN:FERRET.PARM'
	INCLUDE 'FERRET_CMN:PLOT_SETUP.PARM'
	INCLUDE 'FERRET_CMN:XPLOT_SETUP.CMN'
	INCLUDE 'FERRET_CMN:XCONTEXT.CMN'
	INCLUDE 'FERRET_CMN:XTEXT_INFO.CMN'
	INCLUDE 'TMAP_PPLV11INC:AXIS_INC.DECL'	! axis lengths
	INCLUDE 'PPLV11INC:AXIS.INC'	! axis lengths
#endif

* calling argument declarations:
	INTEGER		joint_cx
	CHARACTER	year*4

* internal variable declarations:
	LOGICAL		show_year
	CHARACTER	AX_TITLE*32,
     .			line_buff*255, genl_buff*255
	INTEGER		idim, len, lablen,
     .			nlabs, dset_num, tlen
	REAL 		ul_yspace, ul_labht,  ul_ypos, ul_ysqueeze
	EQUIVALENCE	( ppl_buff, genl_buff )	! save space

* extra upper left label to display the year ?(in case its not on the time axis)
	nlabs = ul_nlabs
	show_year = year .NE. unspecified_name4
	IF ( show_year ) nlabs = nlabs + 1

* need to reduce size of upper left labels ?
	IF     ( nlabs .LE. 2 ) THEN
	   ul_ysqueeze = 1.0
	ELSEIF ( nlabs .EQ. 3 ) THEN
	   ul_ysqueeze = 0.85
	ELSE
	   ul_ysqueeze = 0.65
	ENDIF
	ul_yspace = ul_yspace0 * ul_ysqueeze
	ul_labht  = ul_labht0  * ul_ysqueeze

* define the position labels ("LATITUDE:", "DEPTH:", etc)
* ... stack them above one another in the upper left
	ul_ypos = ul_yorigin + (nlabs-1)*ul_yspace
	DO 300 idim = 1,4
	   IF ( ul_dolab(idim) ) THEN
	      CALL ASCII_LIMITS( joint_cx, idim, line_buff, len )
	         genl_buff = AX_TITLE( idim, cx_grid(joint_cx), tlen )
	      CALL PUT_LABEL( ppl_movlab-idim, ul_xorigin, ul_ypos,
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
	   CALL PUT_LABEL( ppl_movlab-ppl_year, ul_xorigin, ul_ypos,
     .			ppl_left_just,
     .			ul_labht,
     .			'YEAR : '//line_buff(:lablen),
     .			lab_loc_absolute, lab_loc_axlen_plus )
	ENDIF

* data set labels needed ?
	dset_num = cx_data_set( joint_cx )
	IF (  dset_num .NE. unspecified_int4
     .	.AND. dset_num .NE. pdset_irrelevant ) THEN
* ... name of data set
! 5/91	   CALL GET_DSET_NAME( dset_num, line_buff, lablen )
	   CALL GET_SHORT_DSET_NAME( dset_num, line_buff, lablen )
	   CALL PUT_LABEL( ppl_movlab-ppl_dset, 0.0, ul_yorigin,
     .			   ppl_right_just,
     .			   ul_labht0,
     .			   'DATA SET: '//line_buff(:lablen), 
     .			   lab_loc_axlen_plus, lab_loc_axlen_plus )
* ... title of data set
	   CALL GET_DSET_TITLE( dset_num, line_buff, lablen )
	   CALL BOX_LABEL( ppl_movlab-ppl_datitl, line_buff(:lablen),
     .			   0.5, 0.3, 0.75*xlen, 1.5*letsize_dset_title,
     .			   letsize_dset_title, ppl_centered,
     .			   lab_loc_axlen_times, lab_loc_axlen_plus )
* ... DODS URL of data set
	   IF ( ds_des_name(dset_num)(1:7) .EQ. 'http://' ) THEN
	      lablen = INDEX(ds_des_name(dset_num)(8:),'/')
	      lablen = INDEX(ds_des_name(dset_num)(8+lablen:),'/')
     .		     + 7 + lablen
	      CALL PUT_LABEL( ppl_movlab-ppl_dodsurl, 0.0,
     .			   ul_yorigin + 1.1*ul_yspace,
     .			   ppl_right_just,
     .			   0.75*ul_labht0,
     .			   'DODS URL: '//ds_des_name(dset_num)(:lablen), 
     .			   lab_loc_axlen_plus, lab_loc_axlen_plus )
	   ENDIF
	ENDIF

	RETURN
	END
