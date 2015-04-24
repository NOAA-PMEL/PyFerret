	BLOCK DATA pyfont_block_data
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
*	Initialization of pyferret font arguments.
*  pyfont		Flag: True for PyFerret true fonts, False to use Hershey fonts
*  pyfnt_fontname	name of font
*  pyfnt_font_dflt	default font for restoring
*  pyfnt_italic		integer: 1=italic, 0=not
*  pyfnt_bold		integer: 1=bold, 0=not
*  pyfnt_red
*  pyfnt_grn		Real: percent Red, Green, Blue (range 0-100)
*  pyfnt_blu

* These not implemented
*  pyfnt_under		integer: 1=underscore, 0=not
*  pyfnt_opq		Real: percent opacity (range 0-100)
*  pyfnt_rot		Real: rotation angle

	IMPLICIT NONE

	INCLUDE 'pyfonts.cmn'
	INTEGER i

	DATA pyfont/.FALSE./
	DATA igroup/0/
	DATA fnt_all/0/

* index in these structures for each grouping
	DATA itics/1/	! tics
	DATA ititl/2/	! title
	DATA ilogo/3/	! logo
	DATA ihlab/4/	! hlab
	DATA ivlab/5/	! vlab
	DATA icont/6/	! contour
	DATA icbar/7/	! cbar

        DATA    
     .		pyfnt_fontname        /'Hershey'/,	! overall
     .		pyf_fontname(1)       /'Hershey'/,	! tics
     .		pyf_fontname(2)       /'Hershey'/,	! title
     .		pyf_fontname(3)       /'Hershey'/,	! logo
     .		pyf_fontname(4)       /'Hershey'/,	! hlab
     .		pyf_fontname(5)       /'Hershey'/,	! vlab
     .		pyf_fontname(6)       /'Hershey'/,	! contour
     .		pyf_fontname(7)       /'Hershey'/	! cbar

	DATA 
     .		pyfnt_font_dflt        /'Hershey'/,	! overall
     .		pyf_font_dflt(1)       /'Hershey'/,	! tics
     .		pyf_font_dflt(2)       /'Hershey'/,	! title
     .		pyf_font_dflt(3)       /'Hershey'/,	! logo
     .		pyf_font_dflt(4)       /'Hershey'/,	! hlab
     .		pyf_font_dflt(5)       /'Hershey'/,	! vlab
     .		pyf_font_dflt(6)       /'Hershey'/,	! contour
     .		pyf_font_dflt(7)       /'Hershey'/	! cbar

	DATA 
     .		pyf_group_names(1)      /'TICS'/,		! tics
     .		pyf_group_names(2)      /'TITLE'/,		! title
     .		pyf_group_names(3)      /'LOGO'/,		! logo
     .		pyf_group_names(4)      /'HLAB'/,		! hlab
     .		pyf_group_names(5)      /'VLAB'/,		! vlab
     .		pyf_group_names(6)      /'CONTOUR'/,		! contour
     .		pyf_group_names(7)      /'CBAR'/		! cbar

	DATA pyfnt_under/0/
	DATA pyfnt_opq/1./
	DATA pyfnt_rot/0./

	DATA pyfnt_italic/0/
	DATA pyfnt_bold/0/
	DATA pyfnt_red/0./
	DATA pyfnt_blu/0./
	DATA pyfnt_grn/0./
	DATA pyfnt_isiz/0/

	DATA (pyf_italic(i), i=1,ngp)/ngp*0/
	DATA (pyf_bold(i), i=1,ngp)/ngp*0/
	DATA (pyf_red(i), i=1,ngp)/ngp*0./
	DATA (pyf_grn(i), i=1,ngp)/ngp*0./
	DATA (pyf_blu(i), i=1,ngp)/ngp*0./
	DATA (pyf_isiz(i), i=1,ngp)/ngp*0/

* Flag for groups that are set other than the main defaults

	DATA 
     .		pyf_set(1)      /.FALSE./,		! tics
     .		pyf_set(2)      /.FALSE./,		! title
     .		pyf_set(3)      /.FALSE./,		! logo
     .		pyf_set(4)      /.FALSE./,		! hlab
     .		pyf_set(5)      /.FALSE./,		! vlab
     .		pyf_set(6)      /.FALSE./,		! contour
     .		pyf_set(7)      /.FALSE./		! cbar

	END
