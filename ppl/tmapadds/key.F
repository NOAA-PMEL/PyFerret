	SUBROUTINE KEY 

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
*	Generates color/gray scale key
* 	J Davison JISAO/PMEL/NOAA
*	7.20.88

*	Modified to PPL_KEY to fit into ppl
*	J Davison 8.17.88
*	Mod introducing xmaxx and ymaxx to limit size of key JD 4.11.90
*	Mod to control font thickness JD 8.3.90
*	Mod to support ATC individual fill area reps *jd* 4.2.92
*       Mod to support single level chosen by user 8.25.92 *jd*
*	Mod to protect colors used already *jd* 10.1.92
*	Mod to use new hatch scheme *jd* 10.12.92 - cancelled 10.22.92
*	Mod to use SYMBEL rather than GTX for key labels *jd* 3.4.93
* 	Mod to provide identical endpoint values for a constant field/jd/5.96
*       Mod for FILLPOL *jd* 3.99
* v541 *acm*  5/02 add option to control outline on the shade key boxes (cont_key)
*                  also remove VMS INCLUDES
* 552 *acm* 4/03 Shade key improvements: 
* 	1) Larger default label size 
*       2) Can change the location of the labels from right to left and
*           top to bottom, by sending a negative value for the label size.
* 	3) The labels are right-justified (may not look it, depending on font) 
*           for vertical shade keys that are labelled on the left
*       4) Do not have to set all four corners when changing the size and
*           location of the shade key -- can reset any of the four, and
*           others will be filled in by making the key the default size
*           in that direction.  See shade_key.F 
* v553 *acm* Fix bug where length of the first numeric key label was not
*           taken into account when determining label size: computing max_len
* V553 *acm* 9/03 Introduce flag check_0. On Linux, there may be a 
*           be a level of -6.E-07 or some such, where it should be 0.
*           Check whether the numbers for the labels are all small,
*           in which case we leave things alone; otherwise use TM_FPEQ
*           to see if the value should be exactly 0.  
* V554 *acm* 11/03 Call tm_fmt_digits rather than tm_fmt: if the shakey
*                  parameter klab_dig is negative, requesting an exact
*                  number of decimal places.  The change is to force the
*                  same number of digits for all labels. -- more consistent 
*                  output, better justification.
* V580 *acm* 7/04  When the user has set the location and size of the colorbar,
*                  the position of the labels was too close to the edge of the
*                  vertical colorbar.
* V580 *acm* 8/04  Allow for PPL POLYGON as well as PPL FILLPOL after a POLY/SET command.
* V580 *acm* 11/04 Make the default key label size 0.1, to match axis labels. Change
*                  setting so the key labels do not go so close to the edge of the plot.
* V580 *acm* 11/04 Fix bug 906: default number of digits did not distinguish different 
*                  levels; got repeated labels 370, 371, 371, 372, 372,...
* v581 *acm* 3/05 open levels (-INF) (INF)
* V600 *acm* 8/05 fix bug 1330; vertical position of key labels, -0.5*karht was missing 
*                 on definition of yy
* V600 *acm* 9/05 fix bug 1339, with (-inf) (inf) levels and the shakey setting klab_dig=0,
*                 we had an infinite loop on DO 30. Start the loop at ndx = lev1+1, not 1.
* V600 *acm* 1/06 Among the changes when we added tm_fmt_digits, when checking whether more 
*                 digits are needed, keep track of digits added; need to increase the  
*                 argument klab_len by the same amount in subsequent calls to TM_FMT_DIGITS  
*                 when we are writing the numeric labels on the key. 
* V610 *acm* 3/08 For larger default number of levels, make fewer labels when the key is short.

	include 	'parampl5_dat.decl'
	include 	'PARAMPL5.DAT'
	include		'cont_inc.decl'
	include		'CONT.INC'
        include 	'pltcom_dat.decl'
        include 	'PLTCOM.DAT'
        include 	'pltl_inc.decl'
        include 	'PLTL.INC'

        include         'axis_inc.decl'
        include         'AXIS.INC'
        include         'plt_inc.decl'
        include         'PLT.INC'
        include         'pen_inc.decl'
        include         'PEN.INC'

        include         'shade_vars.cmn'
        include         'gkscm1_inc.decl'
        include         'GKSCM1.INC'
        include         'gkscm2.cmn'

	include        'gkspar.inc'

	integer		ndx,str_len,max_len,incr,lab_digits, nhi, ncount,
     .                  lab_add

	real		x_inc,y_inc,px(4),py(4),rem,kpx(5),kpy(5)
	real		xmaxx,ymaxx,vwidth,vheight,karht
	real		xx,yy,xxx,yyy
        real            kx_lo_def, kx_hi_def, ky_lo_def, ky_hi_def,
     .                  val, del, x_inc_inf, y_inc_inf, frac
        real*8          val_n, val_last
	character*12	text,tm_fmt, TM_FMT_DIGITS, txt_out, blanks
        character*20    buff
	logical		TM_FPEQ, TM_DFPEQ, ITSA_AXIS_VIEW, windof_hold, 
     .                  check_0, increase_digits

	external	TM_FMT, TM_FMT_DIGITS

	equivalence 	(px(1),kpx(1)),(py(1),kpy(1))

	include 	'ppl_in_ferret.cmn'	

        DATA      blanks/'            '/

*******************************************************************************

*	CANCEL WINDOW ON IF SET
	windof_hold = windof
	windof = .false.

* Decide whether to check for small numeric values in the key labels.
* If delta is large, we dont want 0.000005 to be labelled in an exponential
* format (it is zero with a bit of noise). But if all the numbers are 
* small, label them as is. 

        del = 1
        check_0 = .TRUE.

        lev1 = 1
        levn = nlev2
        IF (neginf) lev1 = 2
        IF (posinf) levn = nlev2 - 1

        IF (shd_levels .GT. 1) del = zlev(lev1+1) - zlev(lev1)
        IF (TM_FPEQ(del,0.)) check_0 = .FALSE.
        
! IF (INF) levels, will force triangles to be at least 5% of length of colorbar

        frac = 0.05
        ncount = 1./frac

* See if klab_dig is enough digits to distinguish the levels. 

        lab_digits = klab_dig
        lab_add = 0
        increase_digits = .TRUE.

        DO WHILE (increase_digits .AND. ABS(lab_digits) .LE. klab_len )
           buff = ' '

 	   buff = 
     .        TM_FMT_DIGITS (zlev(lev1),lab_digits,klab_len+5,str_len)

           READ (buff,*) val_last
           increase_digits = .FALSE.

	   DO 30 ndx = lev1+1, shd_levels
	     buff = 
     .          TM_FMT_DIGITS (zlev(ndx),lab_digits,klab_len+5,str_len)

              READ (buff,*) val_n
             IF ( TM_DFPEQ(val_last, val_n) ) increase_digits = .TRUE.
             val_last = val_n
   30      CONTINUE

           IF (increase_digits) THEN
              IF (lab_digits .GT. 0) lab_digits = lab_digits + 1
              IF (lab_digits .LT. 0) lab_digits = lab_digits - 1
              lab_add = lab_add + 1
           ENDIF

        ENDDO

	IF (vertical_key) GOTO 1000

*	MAKE A HORIZONTAL KEY
* Define default x and y lo and hi locations

        if (ppl_in_ferret) then
           call get_view_size (vwidth,vheight)

           IF (ITSA_AXIS_VIEW(1)) vheight = (ylen + yorg )* 1.15

           if (vheight .le. ylen + yorg) goto 2000

           rem = vheight - (ylen + yorg)
	   ymaxx = 0.25*ylen
           kx_lo_def = 1000.0 *  xorg
           kx_hi_def = 1000.0 * (xorg + xlen)
           ky_lo_def = 1000.0 * (yorg + ylen + 0.25*min (rem,ymaxx))
           ky_hi_def = 1000.0 * min (yorg+ylen+ymaxx,vheight-0.1*rem)
        else
           rem = height - (ylen + yorg)
	   ymaxx = 0.25*ylen
           kx_lo_def = 1000.0 *  xorg 
           kx_hi_def = 1000.0 * (xorg + xlen)
           ky_lo_def = 1000.0 * (yorg + ylen + 0.25*min (rem,ymaxx))
           ky_hi_def = 1000.0 * min (yorg+ylen+ymaxx,height-0.1*rem)
        end if

        IF (kuser_loc(1) .OR. kuser_loc(2) .OR. 
     .      kuser_loc(3) .OR. kuser_loc(4)) THEN  

! Some or all of kx_lo,kx_hi,ky_lo,ky_hi  are already set

           IF ((.NOT.kuser_loc(1)) .AND. (.NOT.kuser_loc(2))) THEN
              kx_lo = kx_lo_def
              kx_hi = kx_hi_def
           ELSEIF ((.NOT.kuser_loc(1)) .AND. kuser_loc(2) ) THEN
              kx_lo = kx_hi - (kx_hi_def - kx_lo_def)
           ELSEIF (kuser_loc(1) .AND. (.NOT.kuser_loc(2)) ) THEN
              kx_hi = kx_lo + (kx_hi_def - kx_lo_def)
           ENDIF 

           IF ((.NOT.kuser_loc(3)) .AND. (.NOT.kuser_loc(4)) ) THEN
              ky_lo = ky_lo_def
              ky_hi = ky_hi_def
           ELSEIF ((.NOT.kuser_loc(3)) .AND. kuser_loc(4) ) THEN
              ky_lo = ky_hi - 0.4*(ky_hi_def - ky_lo_def)
           ELSEIF (kuser_loc(3) .AND. (.NOT.kuser_loc(4)) ) THEN
              ky_hi = ky_lo + 0.4*(ky_hi_def - ky_lo_def)
           ENDIF 

* When not kuser_loc, we will use the default locations

        ELSE
           kx_lo = kx_lo_def
           kx_hi = kx_hi_def
           ky_lo = ky_lo_def
           ky_hi = ky_hi_def
        ENDIF

	x_inc = (kx_hi - kx_lo)/MAX (1.0, real (shd_levels))
	y_inc = 0.1 * (ky_hi - ky_lo)
        
        x_inc_inf = x_inc

! IF (INF) levels, force triangles to be at least 5% of length of colorbar

        IF (neginf .AND. posinf) THEN
           IF (shd_levels .GT. ncount) THEN 
              x_inc_inf = frac* (kx_hi - kx_lo)
              x_inc = (1.-2.*frac)*(kx_hi - kx_lo)/ REAL(shd_levels-2)
              kx_lo = kx_lo + (x_inc_inf-x_inc)
           ENDIF
        ELSE IF (neginf .OR. posinf) THEN
           IF (shd_levels .GT. ncount) THEN 
              x_inc_inf = frac* (kx_hi - kx_lo)
              x_inc = (1.-frac)*(kx_hi - kx_lo)/ REAL(shd_levels-1)
              IF (neginf) kx_lo = kx_lo + (x_inc_inf-x_inc)
           ENDIF
        ENDIF

*	SET CHARACTER HEIGHT IN WORLD COORDINATES, AND INCR INDEX
	if (klab_incr .eq. 0) then
	   incr = (shd_levels/25) + 1
           IF (width/vwidth .GT. 1.5) incr = 2*incr
	else
	   incr = klab_incr
	end if

	if (klab_size .eq. 0.0) then
           karht = amin1 (0.10*1000.0,
     .		1.0*x_inc*real(incr)/real(max_len))/1000.0
	else
	   karht = ABS(klab_size)
	end if

*	GENERATE THE KEY FOR EACH COLOR/SHADE
	max_len = 0
	text = TM_FMT_DIGITS (zlev(lev1),lab_digits,klab_len+lab_add,str_len)  
        if (max_len .lt. str_len) max_len = str_len
        
        nhi = MAX(2,shd_levels+1)
	do 10 ndx = 2, nhi
	   px(1) = kx_lo + real (ndx - 2) * x_inc
	   px(2) = kx_lo + real (ndx - 2) * x_inc
	   px(3) = kx_lo + real (ndx - 1) * x_inc
	   px(4) = kx_lo + real (ndx - 1) * x_inc

	   py(1) = ky_hi
	   py(2) = ky_hi - 4.0 * y_inc 
	   py(3) = ky_hi - 4.0 * y_inc
	   py(4) = ky_hi

           IF (kuser_loc(3) .OR. kuser_loc(4)) THEN
              py(2) = ky_lo
              py(3) = ky_lo
              py(1) = ky_hi
              py(4) = ky_hi
           ENDIF

           IF (neginf .AND. (ndx .EQ. 2) ) THEN
              CALL key_lo_inf_horz (kpx, kpy, ndx, x_inc_inf, 
     .          annotate_key, lab_digits, karht)

           ELSE IF (posinf .AND. (ndx .EQ. nhi) ) THEN
              CALL key_hi_inf_horz (kpx, kpy, ndx, x_inc_inf, 
     .          annotate_key, lab_digits, karht)

           ELSE

*	   CALL FILL AREA FOR THIS COLOR/GRAY VALUE
	   call set_fill_ndx (ndx)
	   call gfa (4,px,py)

*	   TO RETAIN ABILITY FOR COLOR OR HATCHING IN METAFILE, WRITE 
*	   NDX TO PRIVATE ITEM -- forget it 10.22
*	   if (meta_actv .and. .not. area_bundles) call meta_fill_ndx (ndx)

	   kpx(5) = kpx(1)
	   kpy(5) = kpy(1)
	   IF (.NOT. cont_key) call gpl (5,kpx,kpy)

           ENDIF ! neginf or posinf

           IF (ndx .GE. lev1 .AND. ndx .LE. levn) THEN
              val = zlev(ndx)
              IF (TM_FPEQ(val,0.) .AND. check_0) val = 0.
	      text = TM_FMT_DIGITS (val,lab_digits,klab_len+lab_add,str_len)
	      if (max_len .lt. str_len) max_len = str_len
           ENDIF
10	continue

	do 20 ndx = 1,shd_levels+1,incr
*	   WRITE OUT CORRESPONDING VALUE

CC	   FIX HERE FOR CONSTANT FIELD -- SPECIAL CASE		
	   if (shd_levels .eq. 1 .and. ndx .eq. 2) then
             val = zlev(lev1)
             IF (TM_FPEQ(val,0.) .AND. check_0) val = 0.
	     text = TM_FMT_DIGITS (val,lab_digits,klab_len+lab_add,str_len)

	   else
             val = zlev(ndx)
             IF (TM_FPEQ(val,0.) .AND. check_0) val = 0.
	     text = TM_FMT_DIGITS (val,lab_digits,klab_len+lab_add,str_len)

             IF (ndx .EQ. 1 .AND. neginf) text = " "
             IF (ndx .EQ. shd_levels+1 .AND. posinf) text = " "
	   end if

	   xx = (kx_lo + x_inc*real(ndx-1))/1000.0
     .		 - karht*real(str_len)/2.0 - xorg
	   yy = (ky_hi - 5.0 * y_inc)/1000.0 - yorg - 1.5*karht
           IF (kuser_loc(3) .OR. kuser_loc(4) ) 
     .            yy = (ky_lo - y_inc)/1000.0-yorg-1.5*karht

* If klab_size negative put the key labels on the top rather than the 
* bottom of the key

           IF (klab_size .LT. 0) THEN
              yy = (ky_hi)/1000.0 - yorg + karht
           ENDIF

*	   FILL AND SHADE USE DIFFERENT COORD SYSTEMS -- *jd* 6.4.93
	   if (makep .eq. 'SHADE' .or. makep .eq. 'FILLPOL' .OR.
     .         makep .EQ. 'POLYGON') then
	      call point_convert (xx,yy,xxx,yyy)
	   else if (makep .eq. 'FILL') then
	      xxx = xx
	      yyy = yy
	   endif

	   call symbel (xxx,yyy,0.0,karht,str_len,text(1:str_len))

20	continue

*  Set these to actual locations that were used for PPL LIST SHAKEY
        ky_hi = py(1)
        ky_lo = py(2)

	goto 2000

*	MAKE A VERTICAL KEY 
1000	CONTINUE

* Define default x and y lo and hi locations

        if (ppl_in_ferret) then
           call get_view_size (vwidth,vheight)

           IF (ITSA_AXIS_VIEW(1)) vwidth = (xlen + xorg )* 1.15
           if (vwidth .le. xlen + xorg) goto 2000

	   rem = vwidth - (xlen + xorg)
	   xmaxx = 0.25*xlen

           kx_lo_def = 1000.0 * (xorg + xlen + 0.25*min (rem,xmaxx))
           kx_hi_def = 1000.0 * min (xorg + xlen + xmaxx,vwidth-0.1*rem)
           ky_lo_def = 1000.0 *  yorg
           ky_hi_def = 1000.0 * (yorg + ylen)
        else
	   rem = width - (xlen + xorg)
	   xmaxx = 0.25*xlen
           kx_lo_def = 1000.0 * (xorg + xlen + 0.25*min (rem,xmaxx))
           kx_hi_def = 1000.0 * min (xorg + xlen + xmaxx,width-0.1*rem)
           ky_lo_def = 1000.0 *  yorg
           ky_hi_def = 1000.0 * (yorg + ylen)
        end if

        IF (kuser_loc(1) .OR. kuser_loc(2) .OR. 
     .      kuser_loc(3) .OR. kuser_loc(4)) THEN  

! Some or all of kx_lo,kx_hi,ky_lo,ky_hi  are already set

           IF ((.NOT.kuser_loc(1)) .AND. (.NOT.kuser_loc(2))) THEN
              kx_lo = kx_lo_def
              kx_hi = kx_hi_def
           ELSEIF ((.NOT.kuser_loc(1)) .AND. kuser_loc(2) ) THEN
              kx_lo = kx_hi - 0.4*(kx_hi_def - kx_lo_def)
           ELSEIF (kuser_loc(1) .AND. (.NOT.kuser_loc(2)) ) THEN
              kx_hi = kx_lo + 0.4*(kx_hi_def - kx_lo_def)
           ENDIF 

           IF ((.NOT.kuser_loc(3)) .AND. (.NOT.kuser_loc(4))) THEN
              ky_lo = ky_lo_def
              ky_hi = ky_hi_def
           ELSEIF ((.NOT.kuser_loc(3)) .AND. kuser_loc(4) ) THEN
              ky_lo = ky_hi - (ky_hi_def - ky_lo_def)
           ELSEIF (kuser_loc(3) .AND. (.NOT.kuser_loc(4)) ) THEN
              ky_hi = ky_lo + (ky_hi_def - ky_lo_def)
           ENDIF 

* When not kuser_loc, we will use the default locations

        ELSE
           kx_lo = kx_lo_def
           kx_hi = kx_hi_def
           ky_lo = ky_lo_def
           ky_hi = ky_hi_def
        ENDIF

	x_inc = 0.1 * (kx_hi - kx_lo)
	y_inc = (ky_hi - ky_lo)/MAX (1.0, REAL (shd_levels))

        y_inc_inf = y_inc

! IF (INF) levels, force triangles to be at least 5% of length of colorbar

        IF (neginf .AND. posinf) THEN
           IF (shd_levels .GT. ncount) THEN 
              y_inc_inf = frac* (ky_hi - ky_lo)
              y_inc = (1.-2.*frac)*(ky_hi - ky_lo)/ REAL(shd_levels-2)
              ky_lo = ky_lo + (y_inc_inf-y_inc)
           ENDIF
        ELSE IF (neginf .OR. posinf) THEN
           IF (shd_levels .GT. ncount) THEN 
              y_inc_inf = frac* (ky_hi - ky_lo)
              y_inc = (1.-frac)*(ky_hi - ky_lo)/ REAL(shd_levels-1)
              IF (NEGINF) ky_lo = ky_lo + (y_inc_inf-y_inc)
           ENDIF
        ENDIF

*	SET CHARACTER HEIGHT IN WORLD COORDINATES, AND INCR INDEX
        if (klab_size .eq. 0.0) then
	   karht = amin1 (0.10*1000.0,400.0*rem/real (max_len))/1000.0
	else
	   karht = ABS(klab_size)
	end if

	if (klab_incr .eq. 0) then
	   incr = (shd_levels/25) + 1
           IF (height/vheight .GT. 1.5) incr = 2*incr   ! short plot, fewer labels
	else
	   incr = klab_incr
	end if

*	GENERATE THE KEY FOR EACH COLOR/SHADE
	max_len = 0

	text = TM_FMT_DIGITS (zlev(lev1),lab_digits,klab_len+lab_add,str_len)

	if (max_len .lt. str_len) max_len = str_len	

        nhi = MAX(2,shd_levels+1)
	do 1010 ndx = 2, nhi
	   px(1) = kx_lo
	   px(2) = kx_lo + 4.0*x_inc
	   px(3) = kx_lo + 4.0*x_inc
	   px(4) = kx_lo

           IF ( kuser_loc(1) .OR. kuser_loc(2)) THEN
              px(1) = kx_lo
              px(4) = kx_lo
              px(2) = kx_hi
              px(3) = kx_hi
           ENDIF

	   py(1) = ky_lo + real (ndx - 2) * y_inc
	   py(2) = ky_lo + real (ndx - 2) * y_inc
	   py(3) = ky_lo + real (ndx - 1) * y_inc
	   py(4) = ky_lo + real (ndx - 1) * y_inc


           IF (neginf .AND. (ndx .EQ. 2) ) THEN
              CALL key_lo_inf_vert (kpx, kpy, ndx, y_inc_inf, 
     .          annotate_key, lab_digits, karht)

           ELSE IF (posinf .AND. (ndx .EQ. nhi) ) THEN
              CALL key_hi_inf_vert (kpx, kpy, ndx, y_inc_inf, 
     .          annotate_key, lab_digits, karht)

           ELSE

*	   CALL FILL AREA FOR THIS COLOR/GRAY VALUE
	      call set_fill_ndx (ndx)
	      call gfa (4,px,py)

*	   TO RETAIN ABILITY FOR COLOR OR HATCHING IN METAFILE, WRITE 
*	   NDX TO PRIVATE ITEM - forget it 10.22
*	   if (meta_actv .and. .not. area_bundles) call meta_fill_ndx (ndx)

              kpx(5) = kpx(1)
              kpy(5) = kpy(1)
              IF (.NOT. cont_key) call gpl (5,kpx,kpy)

           ENDIF ! neginf or posinf

           IF (ndx .GE. lev1 .AND. ndx .LE. levn) THEN
              val = zlev(ndx)
              IF (TM_FPEQ(val,0.) .AND. check_0) val = 0.
	      text = TM_FMT_DIGITS (val,lab_digits,klab_len+lab_add,str_len)
	      if (max_len .lt. str_len) max_len = str_len
           ENDIF
1010	continue

        ystart = 0.
        IF (lev1 .NE. 1) ystart = y_inc
        
        do 1020 ndx = lev1,shd_levels+1,incr
*	   WRITE OUT CORRESPONDING VALUE

CC	   FIX HERE FOR CONSTANT FIELD -- SPECIAL CASE		
	   if (shd_levels .eq. 1 .and. ndx .eq. 2) then
             val = zlev(lev1)
             IF (TM_FPEQ(val,0.) .AND. check_0) val = 0.
	     text = TM_FMT_DIGITS (val,lab_digits,klab_len+lab_add,str_len)
	   else
             val = zlev(ndx)
             IF (TM_FPEQ(val,0.) .AND. check_0) val = 0.
	     text = TM_FMT_DIGITS (val,lab_digits,klab_len+lab_add,str_len)
             IF (ndx .EQ. 1 .AND. neginf) text = " "
             IF (ndx .EQ. levn+1 .AND. posinf) text = " "
	   end if

	   xx = (kx_lo + 5.0*x_inc)/1000.0 - xorg 
           IF (kuser_loc(1) .OR. kuser_loc(2)) 
     .              xx = (kx_hi+2.*x_inc)/1000 - xorg

* If klab_size negative put the key labels on the left rather than 
* the right of the key

           IF (klab_size .LT. 0) THEN
	      xx = (kx_lo)/1000.0 - xorg -(REAL(str_len)+1.)* karht
              txt_out = text

* If the key labels are to the right, right-justify them.

           ELSE
              nspace = max_len - str_len
              txt_out = text
              IF (nspace .GT. 0 .AND. del.GT.1)THEN 
                 txt_out = blanks(1:nspace)//text(1:str_len)
                 str_len = max_len
              ENDIF
           ENDIF

	   yy = (ky_lo + y_inc*real (ndx-1))/1000.0 - yorg - 0.5*karht
!	   yy = (ky_lo + y_inc*real (ndx-1))/1000.0 - yorg   (this caused bug 1330!)

*	   FILL AND SHADE USE DIFFERENT COORD SYSTEMS -- *jd* 6.4.93

	   if (makep .eq. 'SHADE' .or. makep .eq. 'FILLPOL' .OR.
     .         makep .EQ. 'POLYGON') then
              call point_convert (xx,yy,xxx,yyy)
	   else if (makep .eq. 'FILL') then
	      xxx = xx
	      yyy = yy
	   endif
	   call symbel (xxx,yyy,0.0,karht,str_len,txt_out(1:str_len))

1020	continue

*  Set these to actual locations that were used for PPL LIST SHAKEY
        kx_lo = px(1)
        kx_hi = px(2)

2000	windof = windof_hold

	return
	end
