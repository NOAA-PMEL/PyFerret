        SUBROUTINE SET_PPL_DEFAULTS(windid)

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
*  ARE DISCLAIMED. IN NO EVENT SHALL NOAA/PMEL/TMAP BE LIABLE FOR ANY 
*  SPECIAL, INDIRECT OR CONSEQUENTIAL DAMAGES OR ANY DAMAGES WHATSOEVER
*  RESULTING FROM LOSS OF USE, DATA OR PROFITS, WHETHER IN AN ACTION OF
*  CONTRACT, NEGLIGENCE OR OTHER TORTUOUS ACTION, ARISING OUT OF OR IN
*  CONNECTION WITH THE ACCESS, USE OR PERFORMANCE OF THIS SOFTWARE. 

*  Using wn_xinches and wn_yinches for windid, 
*  set the rest of ppl values to the scaled default values.

        IMPLICIT NONE

        INCLUDE 'tmap_dims.parm'
        INCLUDE 'ferret.parm'
        INCLUDE 'plot_setup.parm'
        INCLUDE 'xplot_state.cmn'

* argument declaration
        INTEGER windid

* local variables
        REAL pplscale

* compute the scaled default sizes
        pplscale = SQRT( wn_xinches(windid) * wn_yinches(windid)
     .                   / (dflt_xinches * dflt_yinches) )
        wn_pplscale(windid)  = pplscale
        wn_xwhitelo(windid)  = dflt_xwhitelo  * pplscale
        wn_ywhitelo(windid)  = dflt_ywhitelo  * pplscale
        wn_xwhitehi(windid)  = dflt_xwhitehi  * pplscale
        wn_ywhitehi(windid)  = dflt_ywhitehi  * pplscale
        wn_tics_lgx(windid)  = dflt_tics_lg   * pplscale
        wn_tics_lgy(windid)  = dflt_tics_lg   * pplscale
        wn_tics_smx(windid)  = dflt_tics_sm   * pplscale
        wn_tics_smy(windid)  = dflt_tics_sm   * pplscale
        wn_axlsze_x(windid)  = dflt_axlsze    * pplscale
        wn_axlsze_y(windid)  = dflt_axlsze    * pplscale
        wn_txlsze(windid)    = dflt_axlsze    * pplscale
        wn_labset_mn(windid) = dflt_letsiz_lg * pplscale
        wn_labset_x(windid)  = dflt_letsiz_md * pplscale
        wn_labset_y(windid)  = dflt_letsiz_md * pplscale
        wn_labset_mv(windid) = dflt_letsiz_md * pplscale

* and tell PPL about these sizes
        CALL SET_PPL_VALUES(windid)

        RETURN
        END

