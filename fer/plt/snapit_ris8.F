	SUBROUTINE SNAPIT_RIS8 (ws_id,rle,x_pixel_dim,y_pixel_dim,
     .	error_status,ris_file,scan_line,pixmap)

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
*       and store in HDF RIS8 format.  Modification of SNAPIT of yore.

*	Jerry Davison NOAA/PMEL/TMAP					 
*       Mod 3.11.93 for Ferret to capture entire screen when in viewport
* 1/97 *kob* - Linux Port - had to move variable assignments off of
*                           declaration line.
*			  - added check for type byte, because fortran 90 
*                           does not accept this type.

#ifdef unix
        include 'tmap_pplv11inc/pltcom_dat.decl'
        include 'pplv11inc/PLTCOM.DAT'
#else
        INCLUDE 'TMAP_PPLV11INC:PLTCOM_DAT.DECL'
        INCLUDE 'PPLV11INC:PLTCOM.DAT'
#endif

	integer 	ws_id,x_pixel_dim,y_pixel_dim
	integer		scan_line(x_pixel_dim,1),new_trans
	PARAMETER	(new_trans = 10)	
	integer		error_status,invalid_ndx,transno

        integer         rle,d8aimg,i,j
#ifdef NEED_BYTE
        byte            pixmap(x_pixel_dim,y_pixel_dim)
#else
	integer*1	pixmap(x_pixel_dim,y_pixel_dim)
#endif
	real 		world(4),ndc(4),fctx,fcty
	real		x_minw,x_maxw,y_minw,y_maxw,res_y,y_pos

        character*80    ris_file

******************************************************************************

*	GET WORLD COORDINATES AND PIXEL DIMENSIONS OF FRAME
	call  gqcntn (error_status,transno)
	if (error_status .ne. 0) goto 1000

        IF(ASIZE.GT.BSIZE)THEN
           FCTY=BSIZE/ASIZE
           FCTX=1.0
        ELSE
           FCTX=ASIZE/BSIZE
           FCTY=1.0
        ENDIF
	CALL GSWN(new_trans, 0.0, asize, 0.0, bsize )
	CALL GSVP(new_trans, 0.0, fctx,  0.0, fcty )
	call gselnt (new_trans)

        call gqnt (new_trans,error_status,world,ndc)
        if (error_status .ne. 0) goto 1000

        x_minw = world(1)
        x_maxw = world(2)
        y_minw = world(3)
        y_maxw = world(4)

* trial  
* 1	subt .5 to j, add 0 to y_pixel_dim  -> awful!
* 2	add  0 to j,  add 0 to y_pixel_dim  -> 1 scan line missing
* 3	add  0 to j,  add 1 to y_pixel_dim  -> ??

* 	DETERMINE RESOLUTION OF SCREEN, WORLD COORD UNITS/PIXEL
	res_y = (y_maxw - y_minw)/(real(y_pixel_dim+1))

*	DETERMINE THE PIXEL INDICES FOR EACH ROW OF THE PICTURE
	do j = 1,y_pixel_dim

#ifdef atc_gks
	   y_pos = y_maxw - (real(j)-0.5)*res_y
#else
           y_pos = y_maxw - real(j)*res_y
#endif
	   call gqpxa (ws_id,x_minw,y_pos,x_pixel_dim,1,
     .	   1,1,x_pixel_dim,1,error_status,invalid_ndx,scan_line)

	   if (error_status .ne. 0) goto 1000
         
*	   CONVERT EACH ROW TO A BYTE VALUE IN PIXMAP
	   do i = 1,x_pixel_dim
              pixmap(i,j) = scan_line(i,1)
	   end do
	end do

*       WRITE OUT THE FRAME
        error_status = d8aimg (ris_file,pixmap,
     .                         x_pixel_dim,y_pixel_dim,rle)

1000	call gselnt (ws_id)
	return
	end






