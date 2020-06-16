	SUBROUTINE DSG_FEATURE_KEYS (var1, nmv, cx_list, 
     .		key_feature_id, this_no_range, overlay, nokey)
   
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
* Put keys for multi-line plot for a DSG dataset.

	include 'tmap_dims.parm'
	include 'ferret.parm'
	include 'plot_setup.parm'
	include 'xplot_setup.cmn'
	include 'xprog_state.cmn'
	include 'xplot_state.cmn'  
	include 'xcontext.cmn'

* calling argument declarations:
	LOGICAL this_no_range(*), overlay, nokey
	INTEGER var1, nmv, cx_list(*), key_feature_id(*)

* internal variable declarations:
	LOGICAL user_title
	INTEGER	TM_LENSTR1, 
     .		ipl, nline_in_mem, slen, style, 
     .		justify, i
	
	CHARACTER*2040	plot_title
	CHARACTER	TM_FMT*48, buff1*16, fnum*12
	REAL		xkeypos, ykeypos, xkeysize, ykeysize, 
     .			lab_ht, labsize, val


* * * * * * *    - - - TITLES - - -   * * * * * * * * *

	nline_in_mem = 0
	IF ( overlay ) THEN
! TODO...

	ELSE

* ... keys for the dependent vars, or if one, only the title or axis label
	   DO 600 ipl = var1, nmv
	      nline_in_mem = nline_in_mem + 1
	      IF (ipl .GT. max_key_entries) CYCLE
	      IF ( use_keys .AND. (.NOT. nokey) .AND.nmv.GT.var1) THEN
		 val = key_feature_id(ipl)
		 fnum = TM_FMT(val, 8, 10, slen)
		 plot_title = 'Feature '//fnum(:slen)
		 slen = slen + 8
	      ENDIF
              IF (this_no_range(ipl) .AND. mode_nodata_lab) THEN
                 plot_title(slen+1:slen+14) = ' No Valid Data'
                 slen = slen + 14
              ENDIF
	      IF ( nmv .EQ. 1 .OR. user_title ) THEN
	    
	         !?
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

	ENDIF  ! overlay
	
	RETURN
	END
