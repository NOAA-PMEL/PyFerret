	BLOCK DATA XMACHINE_DATA

* variables to assist with the conversion of binary representations
* between differing cpu types
* programmer - steve hankin
* NOAA/PMEL, Seattle, WA - Tropical Modeling and Analysis Program
* written for VAX computer under VMS operating system
*
* SUN port - 1/30/92 -kob-  use xmachine.cmn on SUN because it understands BYTE
*                             and use xmachine.cmn_text for DECstation because
*                             because it understands INTEGER*1
* replaced "elif" syntax with
*	else
*	   if
*  for SGI port	 - kob 4/8/92
* *kob* 10/96   - Linux port.  Linux didn't like tabs between "include"
*                 and what was being included.  removed themy

* revision 0.00 - 03/25/91

#ifdef sun
        include 'tmap_format/xmachine_byte.cmn'
#else
#	if unix
                include 'tmap_format/xmachine_int1.cmn'
#	else
                INCLUDE 'TMAP_FORMAT:XMACHINE_BYTE.CMN'
#	endif
#endif

#ifdef sun
        DATA    active_cpu  /cptype_sun/    
#else
#	if unix
        	DATA	active_cpu  /cptype_dec/   ! DECstation
#	else
        	DATA    active_cpu  /cptype_vax/   ! VAX
#	endif
#endif

	END
