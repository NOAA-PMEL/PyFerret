	INTEGER FUNCTION ALIAS_ID( string )

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
* identify the given string as an alias - retunr unspecified_int4 if not

* programmer - steve hankin
* NOAA/PMEL, Seattle, WA - Tropical Modeling and Analysis Program

* V230:  7/29/92 - from ALIAS_COMMAND
* V320:  8/18/94 - *KOB* IBM port - changed declaration of MATCH4 from	
				    integer to logical
* calling argument declarations:
	CHARACTER string*(*)

* internal variable declarations:
	INTEGER	TM_LENSTR1, TM_LENSTR, alen, i,
     .          len_test, len_mchars
	LOGICAL	MATCH4

#ifdef unix
	include 'ferret.parm'
	include 'xcommand.cmn'
#else
	INCLUDE 'FERRET_CMN:FERRET.PARM'
	INCLUDE 'FERRET_CMN:XCOMMAND.CMN'
#endif

* length of test string
        alen = TM_LENSTR1( string )
        len_test = TM_LENSTR(string(:alen))
* against possible aliases
	DO 100 i = 1, total_num_alias
           IF ( alias(i) .EQ. unspecified_name4 ) GOTO 100
           len_mchars = TM_LENSTR(alias_name(i)(:4))
	   IF ( MATCH4(string(:alen),len_test,
     .          alias_name(i),len_mchars) ) THEN
              ALIAS_ID = i
	      RETURN
	   ENDIF
 100	CONTINUE

* no match
        ALIAS_ID = unspecified_int4
	RETURN
	END
