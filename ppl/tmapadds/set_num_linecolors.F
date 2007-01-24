        SUBROUTINE SET_NUM_LINECOLORS 
        
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
*  5/2006 ACM
*  v6.00 Sets the number of line colors that will be allowed.
* V602: *acm* 1/07 Changes for metafile-batch mode

	INCLUDE 'cmrd_inc.decl'
        INCLUDE 'CMRD.INC'
        INCLUDE 'parampl5_dat.decl'
        INCLUDE 'PARAMPL5.DAT'
        INCLUDE 'plt_inc.decl'
        INCLUDE 'PLT.INC'
        INCLUDE 'pen_inc.decl'
        INCLUDE 'PEN.INC'
        INCLUDE 'gkscm1_inc.decl'            
        INCLUDE 'GKSCM1.INC'     
        INCLUDE 'gkscm2.cmn'                   

******************************************************************************

        INTEGER no_plots_yet

*	If no args, reset to default number of 6 colors
	IF (m .EQ. 0) THEN
           num_line_colors = 6

*	Else set to value in P(1) 
        ELSE IF (p(1) .GE. 6) THEN
           num_line_colors = p(1)
        ELSE 
           num_line_colors = 6
        ENDIF

        no_plots_yet = 0
        no_plots_yet = p(2)

* For metafiles, setup line bundles here so we can 
* save the color definitions.

        IF (gks_open .AND. .NOT.meta_clsplt .AND. no_plots_yet.EQ.1) 
     .      CALL SETUP_LINE_BUNDLES (wsid, wstype)

        RETURN
	END


