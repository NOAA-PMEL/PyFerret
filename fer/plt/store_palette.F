	SUBROUTINE STORE_PALETTE (ws_id)

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
*	Captures current color map in use by GKS ws and stores as 
*	current palette in HDF RIS8 image file.
*	J Davison 1.7.92
*       *jd* && *sh* 3-30-93 - retain indices 1-6 in the raster 8 file
* 1/97 *kob* - Linux Port 
*			  - added check for type byte, because fortran 90 
*                           does not accept this type.

#ifdef unix
        include         'tmap_pplv11inc/gkscm1_inc.decl'
        include         'pplv11inc/GKSCM1.INC'
#else
        INCLUDE         'tmap_pplv11inc:tmap_gkscm1.inc'
#endif

        integer         ws_id,error_status,max_pline,max_pmark
	integer         max_text,max_fill_area,max_pattern,max_color
	integer		i,d8spal
	real		r,g,b
#ifdef NEED_BYTE
	byte		palette (3,0:255)
#else
	integer*1	palette (3,0:255)
#endif
*******************************************************************************

        call gqlwk (wstype,error_status,max_pline,
     .  max_pmark,max_text,max_fill_area,max_pattern,max_color)

        do 10 i = 0,max_color-1
           call gqcr (ws_id,i,1,error_status,r,g,b)
           palette(1,i) = int (255 * r)
           palette(2,i) = int (255 * g)
           palette(3,i) = int (255 * b)
10      end do

* removed 3/93 - retain original colors for indices 1-6
*        call fix_screen (palette)
        error_status = d8spal (palette)

	return
	end

*******************************************************************************

        SUBROUTINE FIX_SCREEN (palette)

*       Fix the top entries in the color table so the screen does not
*       have useless or difficult color scheme for pointers, windows, etc.
* 1/97 *kob* - Linux Port - had to move variable assignments off of
*                           declaration line.
*			  - added check for type byte, because fortran 90 
*                           does not accept this type.

#ifdef NEED_BYTE
        byte       palette(3,0:255)
#else
	integer*1 palette(3,0:255)
#endif

        integer    s1,s2,s3,s4,s5,s6
	parameter (s1 = 1,
     .		   s2 = 2,
     .		   s3 = 3,
     .		   s4 = 4,
     .		   s5 = 5,
     .		   s6 = 6)
***************************************************************************

        palette  (1,s1) = int (.4*255.0)
        palette  (2,s1) = int (.4*255.0)
        palette  (3,s1) = int (.4*255.0)

        palette  (1,s2) = int (.25*255.0)
        palette  (2,s2) = int (.25*255.0)
        palette  (3,s2) = int (.25*255.0)

        palette  (1,s3) = int (255.0)
        palette  (2,s3) = int (255.0)
        palette  (3,s3) = int (.75*255.0)

        palette  (1,s4) = int (.25*255.0)
        palette  (2,s4) = int (0.0)
        palette  (3,s4) = int (0.0)

        palette  (1,s5) = int (.8*255.0)
        palette  (2,s5) = int (.8*255.0)
        palette  (3,s5) = int (.8*255.0)

        palette  (1,s6) = int (255.0)
        palette  (2,s6) = int (.66*255.0)
        palette  (3,s6) = int (0.0)

        return
        end
