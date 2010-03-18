	SUBROUTINE SHOW_DIAG_VARS( name, order )

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
* execute the SHOW VARIABLES/DIAGNOSTIC command - display an alphabetized list 
* of the available variables

* programmer - steve hankin
* NOAA/PMEL, Seattle, WA - Tropical Modeling and Analysis Program
* written for VAX computer under VMS operating system
*
* revision 0.0 - 7/8/87  - from SHOW_MEM_VARS and XEQ_SHOW
* revision 0.1 - 3/25/88 - added interruptibility
* revision 0.2 - 4/5/88  - eliminated cvar_second_name
* revision 0.3 - 4/29/88 - added "name" for name matching
* V200:  7/25/89 - show component vars
* V320: 12/29/94 - use SPLIT_LIST to redirect tty output for GUI
* V650  Write to show_lun which may be set to a file unit number rather than std out

	include	'ferret.parm'
	include 'errmsg.parm'
	include 'xprog_state.cmn'
	include 'xvariables.cmn'
	include	'xcalc_vars.cmn'
	include	'xtext_info.cmn'
	include 'xinterrupt.cmn'

* calling argument declarations:
	INTEGER		order( max_calc_vars )
	CHARACTER	name*4

* local variable declarations:
	LOGICAL MATCH4, all, full
	INTEGER i, i2, temp, len, icomp, comp0, iend, var, status,
     .          TM_LENSTR, len_test, len_mchars
	CHARACTER	var_code*4, comp_buff*64
	DATA		comp_buff(1:9) / '-->using:' /

* local parameters
	INTEGER	    slash_full
	PARAMETER ( slash_full = 1 + 1 )

* initialize
	all  = name .EQ. unspecified_name4
	full = qual_given(slash_full) .GT. 0

* fill the order buffer with the default order
	DO 100 i = 1, num_calc_vars
 100	order(i) = i

* alphabetize the list
	DO 200 i2 = num_calc_vars, 2, -1
	DO 200 i = 2, i2
	   IF ( cvar_name_code(order(i-1)) .GT. cvar_name_code(order(i)) )
     .									THEN
	      temp = order(i)
	      order(i  ) = order(i-1)
	      order(i-1) = temp
	   ENDIF
 200	CONTINUE

* ... and write it out
	   CALL SPLIT_LIST(pttmode_explct, show_lun,
     .		'Diagnostic variables:',21)
	   CALL SPLIT_LIST(pttmode_explct, show_lun,
     .		'name         title',18)
           len_test = TM_LENSTR(name)
	   DO 300 i2 = 1, num_calc_vars
	      IF ( interrupted ) CALL ERRMSG
     .			( ferr_interrupt, status, ' ', *1000 )
	      i = order( i2 )
              len_mchars = TM_LENSTR(cvar_name_code(i)(:4))
	      IF ( all .OR. MATCH4(name,len_test, 
     .            cvar_name_code(i),len_mchars) ) THEN
	         CALL SPLIT_LIST(pttmode_explct, show_lun,
     .			cvar_name_code(i)//'      '//cvar_primary_name(i), 0)
	         IF ( .NOT.full ) GOTO 300
* ... show the components
	         comp0  = cvar_frame_start( i )
	         iend = 9
	         DO 250 icomp = comp0 + 1, comp0 + cvar_frame_length(i)
	            iend = iend + 5
	            var = cv_variable( icomp )
	            IF (var .GT. pcv_spec_flg) var = var - pcv_spec_flg
	            IF ( cv_category( icomp ) .EQ. cat_calc_var ) THEN
	               var_code = cvar_name_code( var )
	            ELSE
	               var_code = fvar_name_code( var )
	            ENDIF
	            WRITE( comp_buff( iend-4:iend ), '(1X,A)' ) var_code
 250	         CONTINUE
	         CALL SPLIT_LIST(pttmode_explct, show_lun, comp_buff, iend)
	      ENDIF
 300	   CONTINUE

 1000	RETURN
	END
