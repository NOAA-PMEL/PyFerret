	SUBROUTINE SET_DSG_ID_LEV (dset, cx, nfeatures, dep_dat, dtype, changed_key)
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
*  derivative works and supporting documentation.  Further, the user5/4/2005
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
* V7.4+ ACM 7/18  DSG file development
* V7.4+ ACM 9/18  Key label intermediate variable does not need plot settings
* V7.51 6/2019 ACM Use Ampersand to separate strings. Only do up to 20 strings
* V7.51 7/2019 ACM Handle cases when the number of values is really large.
* V7.6  3/2020 ACM Need longer strings to store individual levels
*
* for a DSG plot colored by a variable in the feature direction, set the levels 
* and key to plot with levels of 1 to nfeature, and center-labeled color bar. 
* Label the key with the value of the variable along the feature axis, values or 
* strings.  Feature-direction color plots are always treated as discrete by-level
* plots

	include 'tmap_dims.parm'
	include 'ferret.parm'
	include 'plot_setup.parm'
	include 'parampl5_dat.decl'
	include 'PARAMPL5.DAT'
	include 'cont_inc.decl'	! with nlev
	include 'CONT.INC'
	include 'xvariables.cmn'
	include 'shade_vars.cmn'
	include 'xprog_state.cmn'
	include 'slash.parm'

* Argument declarations
	LOGICAL changed_key
	INTEGER	dset, cx, nfeatures, dtype
	REAL    dep_dat(*)

* Internal declarations
	LOGICAL TM_HAS_STRING, its_vertical, fmask(nfeatures), contig
	INTEGER TM_LENSTR1, TM_LENSTR, GET_OFFSET_C_STRING_LEN, 
     .		ndel, slen, alen, ier, ist, i, ic, f1, f2, 
     .		tot_len, ilev, count, status

	PARAMETER (tot_len=1500)
	CHARACTER TM_FMT*128, GET_STRING_ELEMENT*30,
     .		str_buff*128, all_keylab*(tot_len), labsym_buff*30, c*1, num_buff*12


* If the user has set /LEVELS, keep the setting

	IF (qual_given( slash_plot_levels ) .NE. 0) GOTO 5000

* Set up a levels string, /LEVEL=(f1,f2,1) and labels if there aren't too many
* TODO: allow for the mask to be non-contiguous, and set up levels (1)(8)(13)

	CALL MAKE_DSG_FEATURE_MASK(dset, cx, fmask, nfeatures )
	
* If there are too many features to label nicely along the color bar just let it use numbers. 
* make up a  nominal /lev=(1,n_shown,del)


	count = 0
	DO i = 1, nfeatures
	   IF ( fmask(i) ) count = count+1
	ENDDO

	c = "&"
	alen = 0

	f1 = 0
	f2 = 0
	contig = .TRUE.
	ilev = 0

	DO i = 1, nfeatures
	   IF ( fmask(i) ) THEN
	      ilev = ilev+1

	      IF (f1 .EQ. 0) f1 = i
	      IF (f1 .NE. 0) f2 = i
	      IF (i.GT.1) THEN
	         IF (fmask(i) .AND. .NOT.fmask(i-1)) contig = .FALSE.
	      ENDIF

*  Get string value. String them together to send to PPLUS. 
*  (Check for ampersands and replace with a space? Can't make example file 
*  with ampersands in the strings using ncgen, so leaving this out). 

	     IF (count.LT.20) THEN
	     IF (dtype .EQ.  ptype_string ) THEN
		 str_buff =  GET_STRING_ELEMENT(1,nfeatures, 1,1, 1,1, 1,1, 1,1, 1,1, 
     .				dep_dat,
     .				i, 1, 1, 1, 1, 1,
     .				30, slen)
     
c		 DO ic = 1, slen
c		    IF (str_buff(ic:ic) .EQ. c) str_buff(ic:ic) = ' '
c		 ENDDO

	     ELSE 
	        str_buff = TM_FMT(dep_dat(i), 15, 30, slen)

	     ENDIF

	      IF (alen + slen .GT. tot_len) EXIT 
	      all_keylab = all_keylab(:alen)//str_buff(:slen)//c
	      alen = TM_LENSTR1(all_keylab)
	      
	     ENDIF  ! count

	   ENDIF
	ENDDO

	alen = alen - 1  ! don't count last comma

	IF (alen .GT. 0 .AND. count.LT.20)
     .	CALL PPLCMD( from, line, 0, 'CLRKLAB '//all_keylab(:alen ), 1, 1 )

* For large numbers of levels do something nicer like (0,180,5)

	IF (contig) THEN
	   ndel = MAX(1, INT(0.5+FLOAT(count)/40.) )
	   IF (ndel .GT. 2) f1 = 0
	   WRITE (str_buff, 100) f1, f2, ndel

 100	   FORMAT( '(', 2(i3,','), i3, ')' )

* remove spaces (required by parsing in routine LEV !!)
	   slen = TM_LENSTR1(str_buff)
	   ist = 0
	   DO i = 1, slen
	      c = str_buff(i:i)
	      IF (c .NE. ' ') THEN
	         ist=ist+1
	         str_buff(ist:ist) = c
	      ENDIF
	   ENDDO
	   slen = ist

	ELSE  ! discontiguous levels
	   str_buff = ""
	   slen = 0
	   DO i = 1, nfeatures
	      IF ( fmask(i) ) THEN
	         num_buff = TM_FMT(FLOAT(i), 10, 12, ist)
		 str_buff = str_buff(:slen)//'('//num_buff(:ist)//')'
		 slen = TM_LENSTR1(str_buff)
	      ENDIF
	   ENDDO
	         
	ENDIF

	labsym_buff = "LEV_TEXT "//str_buff(:slen)
	slen = TM_LENSTR1(labsym_buff)
	ier = 0
	ist = 0
	CALL SETSYM (labsym_buff, slen, ier, ist)

	slen = TM_LENSTR1(str_buff)
	CALL PPLCMD( from, line, 0, 'LEV,()'//str_buff(:slen ), 1, 1 )

* Set /key=centered
	its_vertical = vertical_key
	CALL PPLCMD (from, line, 0, 'S_KEY 2', 1, 1 )
	changed_key = .TRUE.

	vertical_key = its_vertical

 5000	RETURN
	END
