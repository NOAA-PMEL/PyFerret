        BLOCK DATA XPLOT_SETUP_DATA

*  This software was developed by the Thermal Modeling and Analysis
*  Project(TMAP) of the National Oceanographic and Atmospheric
*  Administration''s (NOAA) Pacific Marine Environmental Lab(PMEL),
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

* revision 0.1 - 9/12/88 - reduced number of defined colors to 10
* V200: 12/21/89
* V540: *sh* 9/01 - added xwhitelo, ywhitelo, ...
* V63  *acm* 10/09 Changes for gfortran build
*      *kms*  2/12 Include tmap_dims.parm for nferdims used by xplot_setup.cmn
* V698 *sh* 3/16 - adjusted key_xorigin and key_ywhite_space
* PyFr *kms*  7/13 Moved key_xspace0, key_yspace0, key_labht0, key_xorigin, 
*                  key_yorigin, key_xwhite_space, key_ywhite_space, 
*                  key_line_frac, ovkey_space, ovkey_yspace, ovkey_labht,
*                  ovkey_xorigin, ovkey_yorigin, ul_labht0, ul_yspace0,
*                  ul_xorigin, and ul_yorigin to plot_setup.parm,
*                  since they are used as parameters, and prefixed with 
*                  dflt_ those that need to be scaled by pplscale or testscale.
* V76 *acm* 11/19 -  Issue 1960: no changes to PyFerret version of this

        IMPLICIT NONE
        include 'tmap_dims.parm'
        include 'ferret.parm'
        include 'xplot_setup.cmn'

        DATA
     .     saved_ppl_yax / .FALSE. /,
     .     animate       / .FALSE. /,
     .     first_frame   / .TRUE. /
	END
