	SUBROUTINE LINE_STYLE( symbol, line_type, line_number, so_far )

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
* line type (dashing), color and symbol into this single call
* Line number ipen will always be associated with line style ipen.
* Prior to this call colors and line types have already been combined via GKS
* bundles to define 6 distinct line styles as bundle indices 1-6
* Symbols are now added to extend this number to 6x17=102
* 1/94: number of line bundles is now num_fancy_lines.  All of these are
* used if /LINES is specified without /SYMBOLS

* programmer - steve hankin
* NOAA/PMEL, Seattle, WA - Tropical Modeling and Analysis Program
* written for VAX computer under VMS operating system

* V200:	  1/2/90
* 	 2/21/90 - changed line style order so all dash patterns are used for
*		   each symbol choice, then the next symbol ...
* 	 5/18/90 - added args symbol and line_type to force styling of lines
* Unix/RISC port - 10/91 - /symb"/noline" -> successive line w/ new symbol
* V230:  7/23/92 - when /LINE_STYLE is given /SYMBOLS must default to OFF
* V301:  1/27/94 - when /LINE_STYLE is given w/out /SYMBOL use num_fancy_lines

#ifdef unix
	include	'ferret_cmn/slash.parm'
	include	'ferret_cmn/plot_setup.parm'
	include	'ferret_cmn/xplot_setup.cmn'
#else
	INCLUDE	'FERRET_CMN:SLASH.PARM'
	INCLUDE	'FERRET_CMN:PLOT_SETUP.PARM'
	INCLUDE	'FERRET_CMN:XPLOT_SETUP.CMN'
#endif

* calling argument declarations
	INTEGER symbol, line_type, line_number, so_far

* internal variable declarations
        LOGICAL unique_symbol, symbol_only1, select_lines
	INTEGER	style, ppl_draw_line, sym_so_far
        SAVE sym_so_far     ! # symbols on the screen ... *sh* 10/91

* internal parameter declarations
	INTEGER	num_line
	PARAMETER     ( num_line = num_plain*num_sym )

* internal variable declarations
	INTEGER	itype, imark, style0, sym(num_sym)
	DATA	sym/1,3,17,19,21,23,25,27,29,31,33,35,39,41,43,47,51/

* initialize
        IF ( so_far .EQ. 1 ) sym_so_far = 0      ! first line on screen
* ... each line needs a new symbol for /SYMBOL or /LINE=n/SYMBOL
        unique_symbol = ( symbol.EQ.qual_on
     .             .AND. (line_type.EQ.qual_off.OR.line_type.GT.0) )
        symbol_only1  = symbol.GT.0 .AND. line_type.EQ.qual_off
	select_lines = symbol.EQ.qual_off .AND. line_type.EQ.qual_on

* respond to qualifiers that force actions
	IF ( line_type .EQ. qual_off ) THEN
	   ppl_draw_line = 3	! PPL LINE command: mark points only
	ELSE
	   ppl_draw_line = 1	! draw the line by default
	ENDIF
	IF ( symbol .EQ. qual_on ) THEN
* ... bypass plain lines
	   style = so_far + num_plain
	ELSE
	   style = so_far		! default: no qualifiers
	ENDIF

* select line type and symbols
        IF ( unique_symbol .OR. symbol_only1 ) THEN 
           itype = 1
           imark = MOD(sym_so_far,num_sym) + 1
           sym_so_far = sym_so_far + 1
	ELSEIF ( select_lines .AND. style .LE. num_fancy_lines ) THEN  ! 1/94
	   itype = style
	   imark = 0
	ELSEIF ( style .LE. num_plain ) THEN
	   itype = style
	   imark = 0
	ELSEIF ( style .LE. num_line ) THEN
	   style0 = (style-num_plain) - 1
	   itype  = MOD(style0,num_plain) + 1
	   imark  = MOD(style0/num_plain,num_sym) + 1
           sym_so_far = MAX( sym_so_far, imark )    ! ... *sh* 10/91
	ELSE
	   itype = 1
	   imark = 0
	ENDIF
	   
* replace selections with forced values if supplied
	IF ( symbol    .GT. 0 ) THEN
	  imark = symbol
	ELSEIF ( symbol .EQ. qual_off ) THEN
          imark = 0                       ! 7/92
	ELSEIF ( imark .NE. 0 ) THEN
	  imark = sym(imark)
	ENDIF
	IF ( line_type .GT. 0 ) itype = line_type

* communicate this to PLOT+
	WRITE ( ppl_buff, 3000 ) line_number, itype
 3000	FORMAT('PEN ',2I4)
	CALL PPLCMD ( from, line, 0, ppl_buff, 1, 1 )
	IF ( imark .GT. 0 ) THEN
* ... "LINE 8 3 1" - symbol number 3 on line 8
	   WRITE ( ppl_buff, 3100 ) line_number, imark, ppl_draw_line
 3100	   FORMAT ('LINE ',3I4)
	ELSE
* ... "LINE 4 1 0" - no symbols on line 4
	   WRITE ( ppl_buff, 3200 ) line_number
 3200	   FORMAT ('LINE ',I4,' 1 0')
	ENDIF
	CALL PPLCMD ( from, line, 0, ppl_buff, 1, 1 )

	RETURN
	END

