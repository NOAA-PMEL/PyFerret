	INTEGER FUNCTION VIEWPORT_NUMBER( view_name )

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
* identify the named viewport

* programmer - steve hankin
* NOAA/PMEL, Seattle, WA - Tropical Modeling and Analysis Program
* written for VAX computer under VMS operating system
*

* calling argument declarations:
	CHARACTER*(*)	view_name

* internal variable declarations:
	LOGICAL	MATCH_NAME
	INTEGER	ivp
        INTEGER TM_LENSTR,len_test,len_mchars

#ifdef unix
	include	'ferret.parm'
	include	'gfdl_vms.parm'
	include 'xplot_state.cmn'
#else
	INCLUDE	'FERRET_CMN:FERRET.PARM'
	INCLUDE	'FERRET_CMN:GFDL_VMS.PARM'
	INCLUDE 'FERRET_CMN:XPLOT_STATE.CMN'
#endif

* revision 0.0 - 8/31/88
* V200: 10/29/89 - viewport 0 not find-able
*	  5/8/90 - bug: MATCH4 --> MATCH_NAME ("OV" doesn't match "OVER")

* search the viewport_names
        len_test = TM_LENSTR(view_name)
	DO 10 ivp = 1, max_viewport
           len_mchars = TM_LENSTR(vp_name(ivp))
	   IF ( MATCH_NAME(view_name, len_test, 
     .        vp_name(ivp),len_mchars) ) THEN
	      VIEWPORT_NUMBER = ivp
	      RETURN
	   ENDIF

 10	CONTINUE

* given VIEWPORT name doesn't match possible options
	VIEWPORT_NUMBER = unspecified_int4
	RETURN

	END
