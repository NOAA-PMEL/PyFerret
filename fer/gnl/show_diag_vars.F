	SUBROUTINE SHOW_DIAG_VARS( name, order )

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

#ifdef unix
	include	'ferret_cmn/ferret.parm'
	include 'ferret_cmn/errmsg.parm'
	include 'ferret_cmn/xprog_state.cmn'
	include 'ferret_cmn/xvariables.cmn'
	include	'ferret_cmn/xcalc_vars.cmn'
	include	'ferret_cmn/xtext_info.cmn'
	include 'ferret_cmn/xinterrupt.cmn'
#else
	INCLUDE	'FERRET_CMN:FERRET.PARM'
	INCLUDE 'FERRET_CMN:ERRMSG.PARM'
	INCLUDE 'FERRET_CMN:XPROG_STATE.CMN'
	INCLUDE 'FERRET_CMN:XVARIABLES.CMN'
	INCLUDE	'FERRET_CMN:XCALC_VARS.CMN'
	INCLUDE	'FERRET_CMN:XTEXT_INFO.CMN'
	INCLUDE 'FERRET_CMN:XINTERRUPT.CMN'
#endif

* calling argument declarations:
	INTEGER		order( max_calc_vars )
	CHARACTER	name*4

* local variable declarations:
	LOGICAL MATCH4, all, full
	INTEGER i, i2, temp, len, icomp, comp0, iend, var, status
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
	   CALL SPLIT_LIST(pttmode_explct, ttout_lun,
     .		'Diagnostic variables:',21)
	   CALL SPLIT_LIST(pttmode_explct, ttout_lun,
     .		'name         title',18)
	   DO 300 i2 = 1, num_calc_vars
	      IF ( interrupted ) CALL ERRMSG
     .			( ferr_interrupt, status, ' ', *1000 )
	      i = order( i2 )
	      IF ( all .OR. MATCH4(name,cvar_name_code(i)) ) THEN
	         CALL SPLIT_LIST(pttmode_explct, ttout_lun,
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
	         CALL SPLIT_LIST(pttmode_explct, ttout_lun, comp_buff, iend)
	      ENDIF
 300	   CONTINUE

 1000	RETURN
	END
