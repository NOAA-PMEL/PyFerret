	LOGICAL FUNCTION ITSA_MODULO_CURVI_AXIS_SUB ( xfield, yfield,
     1    slox_dim1, shix_dim1, slox_dim2, shix_dim2, 
     2    sloy_dim1, shiy_dim1, sloy_dim2, shiy_dim2, dim1_or_2	)

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
*       
*       programmer - jonathan callahan
*       NOAA/PMEL, Seattle, WA - Tropical Modeling and Analysis Program

*       V450: 4/97

*
*       ... ---> DISP_SET_UP ---> LOAD_CURVILINEAR
*               ---> ITSA_MODULO_CURVI_AXIS ---> ITSA_MODULO_CURVI_AXIS_SUB
*       
*       This function checks to see if an an axis is modulo after it has been
*       transformed by the 'curvilinear' transformations.  Curvilinear transformations
*       are invoked by giving Ferret a 3-argument SHADE, or FILL command.
*
*       For an axis to be modulo, in curvilinear coordinates, it must be
*       modulo in normal coordinates AND the transformation must leave the page
*       position of the first point on the axis 'very near' the page position
*       of the last point on the axis.
*
*       Every (x,y) pair (eg. (Lon,Lat)) in the range of data is transformed to
*       a new page position by the curvilinear transofmation.  The transformed
*       page positions (in user coordinates) are stored in the 'xfield' and 
*       'yfield' 2D arrays.  The 'slox_' and 'sloy_' values are used to dimension
*       the arrays.  'dim1_or_2' has a value of 1 or 2 and specifies which page
*       dimension we are looking at.
*
*       To determine 'very near' we will use the 'wwx~' and 'wwy~' values
*       stored in xcurvilinear.cmn.  These contain the X- and Y-axis maximum and minimum
*       page positions (in user coordinates).


#ifdef unix
	include 'ferret_cmn/ferret.parm'
	include	'ferret_cmn/xcontext.cmn'
	include 'ferret_cmn/xcurvilinear.cmn'
#else   
	INCLUDE 'FERRET_CMN:FERRET.PARM'
	INCLUDE	'FERRET_CMN:XCONTEXT.CMN'
#endif  

*       calling argument declarations:
c *kob* 8/97 had to move the continuation marker to col 5 - it was in col 8
c       this bombed aix compiler, though it seems to work on others
	INTEGER slox_dim1, shix_dim1, slox_dim2, shix_dim2, 
     1    sloy_dim1, shiy_dim1, sloy_dim2, shiy_dim2, dim1_or_2
	REAL    xfield( slox_dim1:shix_dim1, slox_dim2:shix_dim2 ),
     1    yfield( sloy_dim1:shiy_dim1, sloy_dim2:shiy_dim2 )

*       internal variable declarations:
	INTEGER	i, imax, number_of_x_cells, number_of_y_cells
	REAL page_width, page_height, minimum_width, minimum_height

*       --- end of introductory code ---

	ITSA_MODULO_CURVI_AXIS_SUB = .TRUE.

*       The number of cells better be the same whether I calculate it from
*       the X or from the Y dimensioning

	number_of_x_cells = shix_dim1 - slox_dim1
	number_of_y_cells = shix_dim2 - slox_dim2


*       An **arbitrary** definition of whats 'very close'

	minimum_width = (wwxmax - wwxmin) / sqrt(real(number_of_x_cells))
	minimum_height = (wwymax - wwymin) / sqrt(real(number_of_y_cells))


*       Now check the horizontal distance between the first and last X indices

*       When dim1 is modulo:
	IF (dim1_or_2 .EQ. 1) THEN

	   DO 100 i = xpos_loss_dim2, xpos_hiss_dim2
	      IF ( ( xfield(xpos_hiss_dim1,i) - xfield(xpos_loss_dim1,i) ) 
     .               .GT. minimum_width ) THEN
		 ITSA_MODULO_CURVI_AXIS_SUB = .FALSE.
		 RETURN
	      ENDIF
 100	   CONTINUE

*       When dim2 is modulo:
	ELSE

	   DO 200 i = xpos_loss_dim1, xpos_hiss_dim1
	      IF ( ( xfield(i,xpos_hiss_dim2) - xfield(i,xpos_loss_dim2) ) 
     .                    .GT. minimum_width ) THEN
		 ITSA_MODULO_CURVI_AXIS_SUB = .FALSE.
		 RETURN
	      ENDIF
 200	   CONTINUE

	ENDIF



*       Now check the vertical distance between the first and last Y indices

*       When dim1 is modulo:
	IF (dim1_or_2 .EQ. 1) THEN

	   DO 300 i = ypos_loss_dim2, ypos_hiss_dim2
	      IF ( ( yfield(ypos_hiss_dim1,i) - yfield(ypos_loss_dim1,i) ) 
     .                  .GT. minimum_height ) THEN
		 ITSA_MODULO_CURVI_AXIS_SUB = .FALSE.
		 RETURN
	      ENDIF
 300	   CONTINUE

*       When dim2 is modulo:
	ELSE

	   DO 400 i = ypos_loss_dim1, ypos_hiss_dim1
	      IF ( ( yfield(i,ypos_hiss_dim2) - yfield(i,ypos_loss_dim2) ) 
     .                   .GT. minimum_height ) THEN
		 ITSA_MODULO_CURVI_AXIS_SUB = .FALSE.
		 RETURN
	      ENDIF
 400	   CONTINUE

	ENDIF

	RETURN
	END 
