        SUBROUTINE SET_WINDOW_SIZE (ws_id,xmin,xmax,ymin,ymax)

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
*      Routine to handle resizing windows -- uses GSWKVP for DEC GKS and 
*      GUESC -5304 in ATC GKS.  ATC doesn't resize the window with GSWKVP.
*      J Davison 3.24.92
*      Mod 8.27.93 to handle XGKS resizing
*      Mod 3.15.94 *jd* to call X event processing 
*      *sh* 3.5.97 Mod to support metafile-only (no X) output through a
*                cuckoo's egg implantation of meta_wsid in place of wsid
*      *acm* 8/27/04 New returns from resize_xgks_window; pixx,pixy which are
*                    the size of the window in pixels.  Save these in common 
*                    gkscm2.cmn, so that in gif and batch modes we can return 
*                    them in symbols PPL$XPIXEL and PPL$YPIXEL.
* V6.8 *acm* 7/9/2012 For batch graphics on RHEL6, change ITS_BATCH_GRAPHICS
*                     from logical to integer, then batch = ITS_BATCH_GRAPHICS().NE.0
*

        IMPLICIT NONE
        INCLUDE 'gkspar.inc'
        INCLUDE 'ws_types.cmn'
        INCLUDE 'gkscm2.cmn'
        INCLUDE 'fgrdel.cmn'

        INTEGER ws_id
        REAL*4  xmin, xmax, ymin, ymax

*       set the window size
        CALL FGD_GSWKVP(ws_id, xmin, xmax, ymin, ymax);
*       pixx and pixy used for PPL$XPIXEL and PPL$YPIXEL
        pixx = NINT((xmax - xmin) * windowdpix(ws_id) * 100.0 / 2.54);
        pixy = NINT((ymax - ymin) * windowdpiy(ws_id) * 100.0 / 2.54);

        RETURN
        END

