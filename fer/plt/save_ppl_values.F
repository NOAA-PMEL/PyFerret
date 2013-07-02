        SUBROUTINE SAVE_PPL_VALUES(windid)

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

*  save the ppl size values for this window in the wn_...() arrays

        IMPLICIT NONE

        INCLUDE 'tmap_dims.parm'
        INCLUDE 'ferret.parm'
        INCLUDE 'plot_setup.parm'
        INCLUDE 'xplot_state.cmn'
        INCLUDE 'parampl5_dat.decl'
        INCLUDE 'PARAMPL5.DAT'
        INCLUDE 'plt_inc.decl'
        INCLUDE 'PLT.INC'         ! width, height
        INCLUDE 'axis_inc.decl'
        INCLUDE 'AXIS.INC'        ! xlen, xcsize, xorg, ylen, ycsize, yorg
        INCLUDE 'tics_inc.decl'
        INCLUDE 'TICS.INC'        ! tsmx, tlx, tsmy, tly
        INCLUDE 'taxis_inc.decl'
        INCLUDE 'TAXIS.INC'       ! tcsize
        INCLUDE 'labcom_inc.decl'
        INCLUDE 'LABCOM.INC'      ! hlab1, hxlab, hylab, hlabs
        INCLUDE 'cont_inc.decl'
        INCLUDE 'CONT.INC'        ! hgt
        INCLUDE 'dashzz_inc.decl'
        INCLUDE 'DASHZZ.INC'      ! dslab, dashln, spacln
        INCLUDE 'vector_inc.decl'
        INCLUDE 'VECTOR.INC'      ! vlen

* argument declaration
        INTEGER windid

        wn_xinches(windid)   = width
        wn_yinches(windid)   = height
        wn_xwhitelo(windid)  = xorg
        wn_ywhitelo(windid)  = yorg
        wn_xwhitehi(windid)  = width - xlen - xorg
        wn_ywhitehi(windid)  = height - ylen - yorg
        wn_tics_smx(windid)  = tsmx
        wn_tics_lgx(windid)  = tlx
        wn_tics_smy(windid)  = tsmy
        wn_tics_lgy(windid)  = tly
        wn_axlsze_x(windid)  = xcsize
        wn_axlsze_y(windid)  = ycsize
        wn_txlsze(windid)    = tcsize
        wn_labset_mn(windid) = hlab1
        wn_labset_x(windid)  = hxlab
        wn_labset_y(windid)  = hxlab
        wn_labset_mv(windid) = hlabs
        wn_con_hgt(windid)   = hgt
        wn_con_labsp(windid) = dslab
        wn_con_dshln(windid) = dashln
        wn_con_spcln(windid) = spacln
        wn_vec_len(windid)   = vlen

        RETURN
        END
