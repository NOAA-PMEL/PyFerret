	INTEGER FUNCTION ALIAS_ID( string )

* identify the given string as an alias - retunr unspecified_int4 if not

* programmer - steve hankin
* NOAA/PMEL, Seattle, WA - Tropical Modeling and Analysis Program

* V230:  7/29/92 - from ALIAS_COMMAND
* V320:  8/18/94 - *KOB* IBM port - changed declaration of MATCH4 from	
				    integer to logical
* calling argument declarations:
	CHARACTER string*(*)

* internal variable declarations:
	INTEGER	TM_LENSTR1, alen, i
	LOGICAL	MATCH4

#ifdef unix
	include 'ferret_cmn/ferret.parm'
	include 'ferret_cmn/xcommand.cmn'
#else
	INCLUDE 'FERRET_CMN:FERRET.PARM'
	INCLUDE 'FERRET_CMN:XCOMMAND.CMN'
#endif

* length of test string
        alen = TM_LENSTR1( string )

* against possible aliases
	DO 100 i = 1, total_num_alias
           IF ( alias(i) .EQ. unspecified_name4 ) GOTO 100
	   IF ( MATCH4(string(:alen),alias_name(i)(:4)) ) THEN
              ALIAS_ID = i
	      RETURN
	   ENDIF
 100	CONTINUE

* no match
        ALIAS_ID = unspecified_int4
	RETURN
	END
