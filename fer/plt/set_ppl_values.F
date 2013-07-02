        SUBROUTINE SET_PPL_VALUES(windid, reset)

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

*  Tell ppl about the size values for this window that were stored in 
*  the wn_...() arrays.  If reset is .TRUE., reset the values in the
*  wn_...() arrays to default values based on the stored xinches and
*  yinches values.

        IMPLICIT NONE

        INCLUDE 'tmap_dims.parm'
        INCLUDE 'ferret.parm'
        INCLUDE 'plot_setup.parm'
        INCLUDE 'xplot_setup.cmn'
        INCLUDE 'xplot_state.cmn'
        INCLUDE 'parampl5_dat.decl'
        INCLUDE 'PARAMPL5.DAT'
        INCLUDE 'cont_inc.decl'
        INCLUDE 'CONT.INC'        ! hgt
        INCLUDE 'dashzz_inc.decl'
        INCLUDE 'DASHZZ.INC'      ! dslab, dashln, spacln
        INCLUDE 'vector_inc.decl'
        INCLUDE 'VECTOR.INC'      ! vlen

* argument declaration
        INTEGER windid
        LOGICAL reset

* local variables
        REAL*4 rone
        CHARACTER*48 buff

        rone = 1.0

        pplscale = SQRT( wn_xinches(windid) * wn_yinches(windid)
     .                   / (dflt_xinches * dflt_yinches) )
        IF ( reset ) THEN
* assign the scaled default sizes
           wn_xwhitelo(windid)  = dflt_xwhitelo   * pplscale
           wn_ywhitelo(windid)  = dflt_ywhitelo   * pplscale
           wn_xwhitehi(windid)  = dflt_xwhitehi   * pplscale
           wn_ywhitehi(windid)  = dflt_ywhitehi   * pplscale
           wn_tics_lgx(windid)  = dflt_tics_lg    * pplscale
           wn_tics_lgy(windid)  = dflt_tics_lg    * pplscale
           wn_tics_smx(windid)  = dflt_tics_sm    * pplscale
           wn_tics_smy(windid)  = dflt_tics_sm    * pplscale
           wn_axlsze_x(windid)  = dflt_axlsze     * pplscale
           wn_axlsze_y(windid)  = dflt_axlsze     * pplscale
           wn_txlsze(windid)    = dflt_axlsze     * pplscale
           wn_labset_mn(windid) = letsize_large   * pplscale
           wn_labset_x(windid)  = letsize_label   * pplscale
           wn_labset_y(windid)  = letsize_label   * pplscale
           wn_labset_mv(windid) = letsize_label   * pplscale
           wn_con_hgt(windid)   = letsize_small   * pplscale
           wn_con_dshln(windid) = dflt_con_dshln  * pplscale
           wn_con_spcln(windid) = dflt_con_dshln  * pplscale
           wn_con_dslab(windid) = dflt_dslab_sp   * pplscale
           wn_vec_len(windid)   = dflt_vec_len    * pplscale
        ENDIF

*       calls PPL SIZE to set width and height (no actual resizing done)
        CALL SET_PLOT_SIZE(wn_xinches(windid), wn_yinches(windid))

*       calls PPL commands AXLEN, ORIGIN, SET VP_WIDTH, SET VP_HEIGHT
        CALL SET_AX_SIZES(rone, rone, 
     .                    wn_xwhitelo(windid), wn_ywhitelo(windid), 
     .                    wn_xwhitehi(windid), wn_ywhitehi(windid))

*       set the tic sizes
        buff = ' '
        IF ( reset ) THEN
           WRITE(buff, 100) wn_tics_smx(windid), wn_tics_lgx(windid), 
     .                      wn_tics_smy(windid), wn_tics_lgy(windid)
 100       FORMAT('TICS',4(',',F7.4),',-1,-1')
        ELSE
           WRITE(buff, 105) wn_tics_smx(windid), wn_tics_lgx(windid), 
     .                      wn_tics_smy(windid), wn_tics_lgy(windid)
 105       FORMAT('TICS',4(',',F7.4))
        ENDIF
        CALL PPLCMD(from, line, 0, buff, 1, 1)

*       set the axis label sizes
        buff = ' '
        WRITE(buff, 110) wn_axlsze_x(windid), wn_axlsze_y(windid)
 110    FORMAT('AXLSZE',2(',',F7.4))
        CALL PPLCMD(from, line, 0, buff, 1, 1)

*       set the time axis label size
        buff = ' '
        WRITE(buff, 120) wn_txlsze(windid)
 120    FORMAT('TXLSZE,',F7.4)
        CALL PPLCMD(from, line, 0, buff, 1, 1)

*       set the label sizes
        buff = ' '
        WRITE(buff, 130) wn_labset_mn(windid), wn_labset_x(windid), 
     .                   wn_labset_y(windid), wn_labset_mv(windid)
 130    FORMAT('LABSET',4(',',F7.4))
        CALL PPLCMD ( from, line, 0, buff, 1, 1 )

*       the following can just be set directly (no extra actions)
        hgt    = wn_con_hgt(windid)
        dashln = wn_con_dshln(windid)
        spacln = wn_con_spcln(windid)
        dslab  = wn_con_dslab(windid)
        vlen   = wn_vec_len(windid)

        RETURN
        END
