	BLOCK DATA XPLOT_SETUP_DATA

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
* initialization of COMMON / XPLOT_SETUP /
* programmer - steve hankin
* NOAA/PMEL, Seattle, WA - Tropical Modeling and Analysis Program
* written for VAX computer under VMS operating system

* revision 0.1 - 9/12/88 - reduced number of defined colors to 10
* V200: 12/21/89

#ifdef unix
	include 'ferret.parm'
	include 'xplot_setup.cmn'
#else
	INCLUDE 'FERRET_CMN:FERRET.PARM'
	INCLUDE 'FERRET_CMN:XPLOT_SETUP.CMN'
#endif

	DATA
     .		saved_ppl_yax	/ .FALSE. /,
     .		key_xspace0	/ 3.0 /,	! nominal X space for keys
     .		key_yspace0	/ 0.25 /,	! nominal Y space for keys
     .		key_labht0	/ 0.12 /,	! nominal label height for keys
     .		key_xorigin	/ 1.5  /,	! leftmost x loc. of keys
     .		key_yorigin	/ -0.85 /,	! y loc. of topmost keys
     .		key_xwhite_space/ 0.3  /,	! x space between keys
     .		key_ywhite_space/ 0.02 /,	! y space between keys
     .		key_line_frac   / 0.3 /,	! how much key for line specimen
     .		ovkey_xspace	/ 2.1 /,
     .		ovkey_yspace	/ 0.18 /,
     .		ovkey_labht	/ 0.08 /,
     .		ovkey_xorigin	/-1.1 /,
     .		ovkey_yorigin	/-0.6 /,
     .		ul_labht0	/0.12 /,
     .		ul_yspace0	/0.2  /,
     .		ul_xorigin	/ 0.0 /,	! measured from Y axis
     .		ul_yorigin	/ 0.6 /		! measured from top x axis

	END
