	BLOCK DATA XCOMMAND_DATA

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
* command names, subcommands and qualifiers
* numbers of subcommands and qualifiers is determined by COMMON/XCOMMAND/
* see XCOMMAND.CMN for instructions on adding new options

* programmer - steve hankin
* NOAA/PMEL, Seattle, WA - Tropical Modeling and Analysis Program
* written for VAX computer under VMS operating system
*
* revision 0.0 - 3/20/86
* V200:   7/5/89 - added CANCEL VARIABLE
*	  9/1/89 - added CANC DATA/ALL,SET VAR and SET DATA/many_options
*	 1/22/90 - added USER command
*	  1/8/90 - added command aliases (LET and FILE)
*	 5/17/90 - added PLOT/SYMBOL and PLOT/LINE
*	  7/6/90 - added LIST/SINGLY
*	 8/29/90 - added LIST/EPKEY=EPIC_key_number
*	 8/31/90 - added SET MOVIE/START=
* Unix/RISC port - 2/91 - cant use "/star" in documentation
*		 - also fixed two doubly defined cells qualifiers()
*		   at nqual_at2o+1 and nqual_at10d+4
* V230: 1/3/92  - added SHOW TRANSFORMS
*       1/27/92 - added aliases QUIT, DATA, and REGION
*        6/8/92 - added CONTOUR/FILL/KEY/NOKEY/LINE
*                       WIRE
*                       DEFINE-SHOW-CANCEL ALIAS
*       6/19/92 - made alias names 8 characters
*       6/26/92 - added /RIGID to SET DATA/FORM=CDF
*        7/6/92 - added /DEPTH to DEFINE AXIS
*       8/25/92 - added /MODULO to DEFINE AXIS
*       11/4/92 - added /UNITS to DEFINE VARIABLE
*               - changed DEFINE VIEW/SIZE/ORIGIN/CLIP to /TEXT/XLIMITS/YLIMITS
*       12/2/92 - added /ZSCALE to the WIRE command
*       12/4/92 - restored /SIZE, /ORIGIN, and /CLIP to DEFINE VIEW
*       1/14/93 - changed SET MOVIE/APPEND to /COMPRESS
*        2/1/93 - added alias ANIMATE
* V300  3/11/93 - added alias SPECTRUM
*       5/24/93 - added qualifier CANC DATA/NOERROR
*       6/16/93 - added QUERY command and 2 new blank commands
* V301  10/22/93 - added /XLIMITS and /YLIMITS to PLOT command
*	11/03/93 - added DEFINE AXIS/NPOINTS
*	11/05/93 - changed QUERY/PIPE to QUERY/IGNORE (a NO-OP)
*	11/23/93 - added USER/FILE
*	12/8/93  - added VECTOR/XSKIP/YSKIP
*	 1/19/94 - added alias LABEL
*		 - and USER/FORMAT
*	 2/10/94 - added CONTOTUR/PEN, VECTOR/PEN, and LET/QUIET
* V310   4/11/94 - eliminated qualifier LIST/EPKEY -- use SAVE instead
*	    5/94 - added SET DATA/ORDER
* V320   4/11/94 - added SET MEMORY/SIZE=
*	 8/03/94 - added FRAME/FORMAT= and FRAME/FILE=
*	 8/05/94 - added SAVE/{XYZTIJKL}LIMITS=lo:hi
*	10/28/94 - added GO/HELP
*	 1/25/94 - added SAVE/TITLE="my title"
* V400	  2/6/95 - added SHOW SYMBOL
* V402	    7/95 - added DEFINE VAR/D, SHOW VAR/D, CANC VAR/D, SET VAR/D
* V420	 9/22/95 - added SHOW MEMORY/GRIDS
*	 9/29/95 - added LIST/precision=
*	 2/7/96  - added STAT/BRIEF
*	 4/23/96 - increased number of aliases from 30 to 150
* V450	11/14/96 - added LIST/CLOBBER
*	5/97	 - added /NOAXES to SHADE, PLOT, CONTOUR, and VECTOR
*       7/97     - added SHOW FUNCTIONS
*  *js* 7/98     - added SET DATA/TYPE and SET DATA/SWAP
* v491  12.98    - added PATTERN support *jd*
* v491  12.98    - added PATTERN alias *jd*
* V500  2/99 - *kob* - add /QUIET to LIST command
*       2/99 *sh* - added POLYGON command
*       3/99 *jd* - more qualifs for POLYGON and fixed num_qualifs bug for it
*       3/99 *jc* - added /INTERNAL to SHOW FUNCTION command
*       4/99 *sh* - added PAUSE alias
*	     *sh* - added DEFINE AXIS/EDGES and SHOW AXIS/X/Y/Z/T/IJ/K/L 
*	     *sh* - added SET AXIS/DEPTH
*	6/99 *sh* - added WHERE alias
* V510  3/00 *sh* - added SET DATA/REGULAR_T
*		  - added PLOT/COLOR=/THICKNESS=/SIZE=
*		  - added POLYGON/COLOR=/THICKNESS=
*		  - changed /PEN to /COLOR on CONTOUR and VECTOR
*		  - added VECTOR/THICK
*	5/00 *sh* - added DEFINE VARIABLE/BAD=
*	5/00 *sh* - added CANCEL GRID
*	5/00 *sh* - added /HLIMITS and /VLIMITS to graphics commands
*		  - added /DETAILED to SHOW FUNCTIONS and to MESSAGE
*	5/00 *sh* - restored /PEN as a "denigrated" command
* V520: 6/00 *sh* - added MESSAGE/ERROR
* V530: 10/00 *sh* - added added SET DATA/EZ  /TYPE=/DELIMITERS=
*      10/00 *AM* - added VECTOR/STREAMLINE (alias STREAMLINE)

#ifdef unix
	include 'ferret.parm'
	include 'xcommand.cmn'
#else
	INCLUDE 'FERRET_CMN:FERRET.PARM'
	INCLUDE 'FERRET_CMN:XCOMMAND.CMN'
#endif

* local variable declarations:
        INTEGER i

* SET	 - command # 1
	DATA commands(1)			/ 'SET '	/
	DATA num_subcommands(1)			/ nsub1		/
	DATA subcommand_pointer(1)		/ nsub_at1	/
* SET '    '
	   DATA subcommands      (nsub_at1 + 0)	/ '    '	/
	   DATA num_qualifs      (nsub_at1 + 0)	/ nqual1a	/
	   DATA qualifier_pointer(nsub_at1 + 0)	/ nqual_at1a	/
* SET WINDOW
	   DATA subcommands      (nsub_at1 + 1)	/ 'WINDOW'	/
	   DATA num_qualifs      (nsub_at1 + 1)	/ nqual1b	/
	   DATA qualifier_pointer(nsub_at1 + 1)	/ nqual_at1b	/
* SET WINDOW/SIZE...
		DATA qualifiers  (nqual_at1b+0)	/ 'SIZE'	/
		DATA qualifiers  (nqual_at1b+1)	/ 'NEW' 	/
		DATA qualifiers  (nqual_at1b+2)	/ 'LOCATION'	/
		DATA qualifiers  (nqual_at1b+3)	/ 'ASPECT'	/
		DATA qualifiers  (nqual_at1b+4)	/ 'CLEAR'	/
		DATA qualifiers  (nqual_at1b+5)	/ '****'	/
* SET REGION
	   DATA subcommands      (nsub_at1 + 2)	/ 'REGION'	/
	   DATA num_qualifs      (nsub_at1 + 2)	/ nqual1c	/
	   DATA qualifier_pointer(nsub_at1 + 2)	/ nqual_at1c	/
* SET REGION/I/J/K/L/X/Y/Z/T
		DATA qualifiers  (nqual_at1c+0)	/ 'I'	/
		DATA qualifiers  (nqual_at1c+1)	/ 'J'	/
		DATA qualifiers  (nqual_at1c+2)	/ 'K'	/
		DATA qualifiers  (nqual_at1c+3)	/ 'L'	/
		DATA qualifiers  (nqual_at1c+4)	/ 'X'	/
		DATA qualifiers  (nqual_at1c+5)	/ 'Y'	/
		DATA qualifiers  (nqual_at1c+6)	/ 'Z'	/
		DATA qualifiers  (nqual_at1c+7)	/ 'T'	/
		DATA qualifiers  (nqual_at1c+8) / '****'/
		DATA qualifiers  (nqual_at1c+9) / '****'/
		DATA qualifiers  (nqual_at1c+10)/ 'DX'/
		DATA qualifiers  (nqual_at1c+11)/ 'DY'/
		DATA qualifiers  (nqual_at1c+12)/ 'DZ'/
		DATA qualifiers  (nqual_at1c+13)/ 'DT'/
		DATA qualifiers  (nqual_at1c+14)/ 'DI'/
		DATA qualifiers  (nqual_at1c+15)/ 'DJ'/
		DATA qualifiers  (nqual_at1c+16)/ 'DK'/
		DATA qualifiers  (nqual_at1c+17)/ 'DL'/
* SET VIEWPORT
	   DATA subcommands      (nsub_at1 + 3)	/ 'VIEWPORT'	/
	   DATA num_qualifs      (nsub_at1 + 3)	/ nqual1d	/
	   DATA qualifier_pointer(nsub_at1 + 3)	/ nqual_at1d	/
* SET VIEWPORT/SIZE/ORIGIN/CLIP
		DATA qualifiers  (nqual_at1d+0)	/ '****'	/
* SET EXPRSIONION
	   DATA subcommands      (nsub_at1 + 4)	/ 'EXPRSION'	/
	   DATA num_qualifs      (nsub_at1 + 4)	/ nqual1e	/
	   DATA qualifier_pointer(nsub_at1 + 4)	/ nqual_at1e	/
* SET LIST
	   DATA subcommands      (nsub_at1 + 5)	/ 'LIST'	/
	   DATA num_qualifs      (nsub_at1 + 5)	/ nqual1f	/
	   DATA qualifier_pointer(nsub_at1 + 5)	/ nqual_at1f	/
* SET LIST/PRECISION=#digits, /FILE=filename, /FORMAT=list_format, /APPEND
		DATA qualifiers  (nqual_at1f+0)	/ 'PRECISON'	/
		DATA qualifiers  (nqual_at1f+1)	/ 'FILE'	/
		DATA qualifiers  (nqual_at1f+2)	/ 'FORMAT'	/
		DATA qualifiers  (nqual_at1f+3)	/ 'APPEND'	/
		DATA qualifiers  (nqual_at1f+4)	/ 'HEADING'	/
* SET DATA_SET
	   DATA subcommands      (nsub_at1 + 6)	/ 'DATA'	/
	   DATA num_qualifs      (nsub_at1 + 6)	/ nqual1g	/
	   DATA qualifier_pointer(nsub_at1 + 6)	/ nqual_at1g	/
* SET DATA/FORM/GRID/TITL/SIZE/TYPE
		DATA qualifiers  (nqual_at1g+0)	/ 'EZ'  	/
		DATA qualifiers  (nqual_at1g+1)	/ 'VARIABLE'	/
		DATA qualifiers  (nqual_at1g+2)	/ 'TITLE'	/
		DATA qualifiers  (nqual_at1g+3)	/ 'FORMAT'	/
		DATA qualifiers  (nqual_at1g+4)	/ 'GRID'	/
		DATA qualifiers  (nqual_at1g+5)	/ 'SKIP'	/
		DATA qualifiers  (nqual_at1g+6)	/ 'COLUMNS'	/
		DATA qualifiers  (nqual_at1g+7)	/ 'SAVE'	/
		DATA qualifiers  (nqual_at1g+8)	/ 'RESTORE'	/
		DATA qualifiers  (nqual_at1g+9)	/ 'ORDER'	/   !5/94
		DATA qualifiers  (nqual_at1g+10) / 'TYPE'       / 
		DATA qualifiers  (nqual_at1g+11) / 'SWAP'       / 
		DATA qualifiers  (nqual_at1g+12) / 'REGULART'   /
		DATA qualifiers  (nqual_at1g+13) / 'DELIMITER' /
		DATA qualifiers  (nqual_at1g+14) / '****' /  ! available
* SET MODE
	   DATA subcommands      (nsub_at1 + 7)	/ 'MODE'	/
	   DATA num_qualifs      (nsub_at1 + 7)	/ nqual1h	/
	   DATA qualifier_pointer(nsub_at1 + 7)	/ nqual_at1h	/
* SET MODE/LAST
		DATA qualifiers  (nqual_at1h+0)	/ 'LAST'	/
* SET MOVIE
	   DATA subcommands      (nsub_at1 + 8)	/ 'MOVI'	/
	   DATA num_qualifs      (nsub_at1 + 8)	/ nqual1i	/
	   DATA qualifier_pointer(nsub_at1 + 8)	/ nqual_at1i	/
* SET MOVIE/FILE=filename/COMPRESS/LASER
		DATA qualifiers  (nqual_at1i+0)	/ 'FILE'	/
		DATA qualifiers  (nqual_at1i+1)	/ 'COMPRESS'	/
		DATA qualifiers  (nqual_at1i+2)	/ 'LASER'	/
		DATA qualifiers  (nqual_at1i+3)	/ 'START'	/
		DATA qualifiers  (nqual_at1i+4)	/ '****'	/
* SET VARIABLE
	   DATA subcommands      (nsub_at1 + 9)	/ 'VARIABLE'	/
	   DATA num_qualifs      (nsub_at1 + 9)	/ nqual1j	/
	   DATA qualifier_pointer(nsub_at1 + 9)	/ nqual_at1j	/
* SET VARIABLE/TITL
		DATA qualifiers  (nqual_at1j+0)	/ 'TITLE'	/
		DATA qualifiers  (nqual_at1j+1)	/ 'UNIT'	/
		DATA qualifiers  (nqual_at1j+2)	/ 'GRID'	/
		DATA qualifiers  (nqual_at1j+3)	/ 'BAD'	/
		DATA qualifiers  (nqual_at1j+4)	/ 'DATASET'	/
		DATA qualifiers  (nqual_at1j+5)	/ '****'	/
		DATA qualifiers  (nqual_at1j+6)	/ '****'	/
* SET GRID
	   DATA subcommands      (nsub_at1 +10)	/ 'GRID'	/
	   DATA num_qualifs      (nsub_at1 +10)	/ nqual1k	/
	   DATA qualifier_pointer(nsub_at1 +10)	/ nqual_at1k	/
* SET GRID
		DATA qualifiers  (nqual_at1k+0)	/ 'SAVE'	/
		DATA qualifiers  (nqual_at1k+1)	/ 'RESTORE'	/
* SET AXIS
	   DATA subcommands      (nsub_at1 +11)	/ 'AXIS'	/
	   DATA num_qualifs      (nsub_at1 +11)	/ nqual1l	/
	   DATA qualifier_pointer(nsub_at1 +11)	/ nqual_at1l	/
* SET AXIS / MODULO
		DATA qualifiers  (nqual_at1l+0)	/ 'MODULO'	/
		DATA qualifiers  (nqual_at1l+1)	/ 'DEPTH'	/
* SET MEMO							
	   DATA subcommands      (nsub_at1 +12)	/ 'MEMORY'	/
	   DATA num_qualifs      (nsub_at1 +12)	/ nqual1m	/
	   DATA qualifier_pointer(nsub_at1 +12)	/ nqual_at1m	/
* SET MEMO / MWORDS
		DATA qualifiers  (nqual_at1m+0)	/ 'SIZE'	/
		DATA qualifiers  (nqual_at1m+1)	/ '****'	/
* SET ****							  ! unused (n)
	   DATA subcommands      (nsub_at1 +13)	/ '****'	/
	   DATA num_qualifs      (nsub_at1 +13)	/ nqual1n	/
	   DATA qualifier_pointer(nsub_at1 +13)	/ nqual_at1n	/
* SET **** / ****
		DATA qualifiers  (nqual_at1n+0)	/ '****'	/
		DATA qualifiers  (nqual_at1n+1)	/ '****'	/
* SET ****							  ! unused (o)
	   DATA subcommands      (nsub_at1 +14)	/ '****'	/
	   DATA num_qualifs      (nsub_at1 +14)	/ nqual1o	/
	   DATA qualifier_pointer(nsub_at1 +14)	/ nqual_at1o	/
* SET **** / ****
		DATA qualifiers  (nqual_at1o+0)	/ '****'	/
		DATA qualifiers  (nqual_at1o+1)	/ '****'	/
* SET ****							  ! unused (p)
	   DATA subcommands      (nsub_at1 +15)	/ '****'	/
	   DATA num_qualifs      (nsub_at1 +15)	/ nqual1p	/
	   DATA qualifier_pointer(nsub_at1 +15)	/ nqual_at1p	/
* SET **** / ****
		DATA qualifiers  (nqual_at1p+0)	/ '****'	/
		DATA qualifiers  (nqual_at1p+1)	/ '****'	/
* SET ****							  ! unused (q)
	   DATA subcommands      (nsub_at1 +16)	/ '****'	/
	   DATA num_qualifs      (nsub_at1 +16)	/ nqual1q	/
	   DATA qualifier_pointer(nsub_at1 +16)	/ nqual_at1q	/
* SET **** / ****
		DATA qualifiers  (nqual_at1q+0)	/ '****'	/
		DATA qualifiers  (nqual_at1q+1)	/ '****'	/
* SET ****							  ! unused (r)
	   DATA subcommands      (nsub_at1 +17)	/ '****'	/
	   DATA num_qualifs      (nsub_at1 +17)	/ nqual1r	/
	   DATA qualifier_pointer(nsub_at1 +17)	/ nqual_at1s	/
* SET **** / ****
		DATA qualifiers  (nqual_at1r+0)	/ '****'	/
		DATA qualifiers  (nqual_at1r+1)	/ '****'	/
* SET ****							  ! unused (s)
	   DATA subcommands      (nsub_at1 +18)	/ '****'	/
	   DATA num_qualifs      (nsub_at1 +18)	/ nqual1s	/
	   DATA qualifier_pointer(nsub_at1 +18)	/ nqual_at1s	/
* SET **** / ****
		DATA qualifiers  (nqual_at1s+0)	/ '****'	/
		DATA qualifiers  (nqual_at1s+1)	/ '****'	/
		DATA qualifiers  (nqual_at1s+2)	/ '****'	/
		DATA qualifiers  (nqual_at1s+3)	/ '****'	/
* SET ****							  ! unused (t)
	   DATA subcommands      (nsub_at1 +19)	/ '****'	/
	   DATA num_qualifs      (nsub_at1 +19)	/ nqual1t	/
	   DATA qualifier_pointer(nsub_at1 +19)	/ nqual_at1t	/
* SET **** / ****
		DATA qualifiers  (nqual_at1t+0)	/ '****'	/
		DATA qualifiers  (nqual_at1t+1)	/ '****'	/
		DATA qualifiers  (nqual_at1t+2)	/ '****'	/
		DATA qualifiers  (nqual_at1t+3)	/ '****'	/

* SHOW	 - command # 2
	DATA commands(2)			/ 'SHOW '	/
	DATA num_subcommands(2)			/ nsub2		/
	DATA subcommand_pointer(2)		/ nsub_at2	/
* SHOW '    '
	   DATA subcommands      (nsub_at2 + 0)	/ '    '	/
	   DATA num_qualifs      (nsub_at2 + 0)	/ nqual2a	/
	   DATA qualifier_pointer(nsub_at2 + 0)	/ nqual_at2a	/
* SHOW /ALL
		DATA qualifiers  (nqual_at2a+0)	/ 'ALL '	/
* SHOW WINDOWS
	   DATA subcommands      (nsub_at2 + 1)	/ 'WINDOW'	/
	   DATA num_qualifs      (nsub_at2 + 1)	/ nqual2b	/
	   DATA qualifier_pointer(nsub_at2 + 1)	/ nqual_at2b	/
* SHOW WINDOWS /ALL
		DATA qualifiers  (nqual_at2b+0)	/ 'ALL '	/
* SHOW REGION
	   DATA subcommands      (nsub_at2 + 2)	/ 'REGION'	/
	   DATA num_qualifs      (nsub_at2 + 2)	/ nqual2c	/
	   DATA qualifier_pointer(nsub_at2 + 2)	/ nqual_at2c	/
* SHOW REGIONS /ALL
		DATA qualifiers  (nqual_at2c+0)	/ 'ALL '	/
* SHOW AXIS
	   DATA subcommands      (nsub_at2 + 3)	/ 'AXIS'	/
	   DATA num_qualifs      (nsub_at2 + 3)	/ nqual2d	/
	   DATA qualifier_pointer(nsub_at2 + 3)	/ nqual_at2d	/
* SHOW AXIS /ALL
		DATA qualifiers  (nqual_at2d+0)	/ 'ALL '	/
		DATA qualifiers  (nqual_at2d+1)	/ 'I'	/
		DATA qualifiers  (nqual_at2d+2)	/ 'J'	/
		DATA qualifiers  (nqual_at2d+3)	/ 'K'	/
		DATA qualifiers  (nqual_at2d+4)	/ 'L'	/
		DATA qualifiers  (nqual_at2d+5)	/ 'X'	/
		DATA qualifiers  (nqual_at2d+6)	/ 'Y'	/
		DATA qualifiers  (nqual_at2d+7)	/ 'Z'	/
		DATA qualifiers  (nqual_at2d+8)	/ 'T'	/
		DATA qualifiers  (nqual_at2d+9)	/ '****'/
		DATA qualifiers  (nqual_at2d+10)/ '****'/
		DATA qualifiers  (nqual_at2d+11)/ '****'/
* SHOW EXPRSIONION
	   DATA subcommands      (nsub_at2 + 4)	/ 'EXPRSION'	/
	   DATA num_qualifs      (nsub_at2 + 4)	/ nqual2e	/
	   DATA qualifier_pointer(nsub_at2 + 4)	/ nqual_at2e	/
* SHOW EXPRSIONION /ALL
		DATA qualifiers  (nqual_at2e+0)	/ 'ALL '	/
* SHOW LIST
	   DATA subcommands      (nsub_at2 + 5)	/ 'LIST'	/
	   DATA num_qualifs      (nsub_at2 + 5)	/ nqual2f	/
	   DATA qualifier_pointer(nsub_at2 + 5)	/ nqual_at2f	/
* SHOW LIST /ALL
		DATA qualifiers  (nqual_at2f+0)	/ 'ALL '	/
* SHOW DATA_SET
	   DATA subcommands      (nsub_at2 + 6)	/ 'DATA'	/
	   DATA num_qualifs      (nsub_at2 + 6)	/ nqual2g	/
	   DATA qualifier_pointer(nsub_at2 + 6)	/ nqual_at2g	/
* SHOW DATA_SET /ALL/BRIEF
		DATA qualifiers  (nqual_at2g+0)	/ 'ALL '	/
		DATA qualifiers  (nqual_at2g+1)	/ 'BRIEF'	/
		DATA qualifiers  (nqual_at2g+2)	/ 'FULL'	/
		DATA qualifiers  (nqual_at2g+3)	/ 'VARIABLE'	/
		DATA qualifiers  (nqual_at2g+4)	/ 'FILE'	/
		DATA qualifiers  (nqual_at2g+5)	/ '****'	/
* SHOW MODE
	   DATA subcommands      (nsub_at2 + 7)	/ 'MODE'	/
	   DATA num_qualifs      (nsub_at2 + 7)	/ nqual2h	/
	   DATA qualifier_pointer(nsub_at2 + 7)	/ nqual_at2h	/
* SHOW MODE /ALL
		DATA qualifiers  (nqual_at2h+0)	/ 'ALL '	/
* SHOW MOVIE
	   DATA subcommands      (nsub_at2 + 8)	/ 'MOVIE'	/
	   DATA num_qualifs      (nsub_at2 + 8)	/ nqual2i	/
	   DATA qualifier_pointer(nsub_at2 + 8)	/ nqual_at2i	/
* SHOW MOVIE /ALL
		DATA qualifiers  (nqual_at2i+0)	/ 'ALL '	/
* SHOW VARIABLES
	   DATA subcommands      (nsub_at2 + 9)	/ 'VARIABLE'	/
	   DATA num_qualifs      (nsub_at2 + 9)	/ nqual2j	/
	   DATA qualifier_pointer(nsub_at2 + 9)	/ nqual_at2j	/
* SHOW VARIABLES /ALL /DIAGNOSTIC /USER
		DATA qualifiers  (nqual_at2j+0)	/ 'ALL '	/
		DATA qualifiers  (nqual_at2j+1)	/ 'DATASET'	/
		DATA qualifiers  (nqual_at2j+2)	/ 'DIAG'	/
		DATA qualifiers  (nqual_at2j+3)	/ 'USER'	/
		DATA qualifiers  (nqual_at2j+4)	/ '****'	/
* SHOW COMMANDS
	   DATA subcommands      (nsub_at2 +10)	/ 'COMMANDS'	/
	   DATA num_qualifs      (nsub_at2 +10)	/ nqual2k	/
	   DATA qualifier_pointer(nsub_at2 +10)	/ nqual_at2k	/
* SHOW COMMANDS /ALL
		DATA qualifiers  (nqual_at2k+0)	/ 'ALL '	/
* SHOW MEMORY
	   DATA subcommands      (nsub_at2 +11)	/ 'MEMORY'	/
	   DATA num_qualifs      (nsub_at2 +11)	/ nqual2l	/
	   DATA qualifier_pointer(nsub_at2 +11)	/ nqual_at2l	/
* SHOW MEMORY /ALL /TEMP /PERM /FREE
		DATA qualifiers  (nqual_at2l+0)	/ 'ALL '	/
		DATA qualifiers  (nqual_at2l+1)	/ '****'	/
		DATA qualifiers  (nqual_at2l+2)	/ 'TEMPORY'	/
		DATA qualifiers  (nqual_at2l+3)	/ 'PERMANT'	/
		DATA qualifiers  (nqual_at2l+4)	/ 'FREE'	/
* SHOW GRID
	   DATA subcommands      (nsub_at2 +12)	/ 'GRID'	/
	   DATA num_qualifs      (nsub_at2 +12)	/ nqual2m	/
	   DATA qualifier_pointer(nsub_at2 +12)	/ nqual_at2m	/
* SHOW GRID /ALL/I/J/K/L/X/Y/Z/T/D/ ****
		DATA qualifiers  (nqual_at2m+0)	/ 'ALL '	/
		DATA qualifiers  (nqual_at2m+1)	/ 'I   '	/
		DATA qualifiers  (nqual_at2m+2)	/ 'J   '	/
		DATA qualifiers  (nqual_at2m+3)	/ 'K   '	/
		DATA qualifiers  (nqual_at2m+4)	/ 'L   '	/
		DATA qualifiers  (nqual_at2m+5)	/ 'X   '	/
		DATA qualifiers  (nqual_at2m+6)	/ 'Y   '	/
		DATA qualifiers  (nqual_at2m+7)	/ 'Z   '	/
		DATA qualifiers  (nqual_at2m+8)	/ 'T   '	/
		DATA qualifiers  (nqual_at2m+9)	/ 'DYNAMIC'	/
		DATA qualifiers  (nqual_at2m+10)/ '****'	/
* SHOW VIEW
	   DATA subcommands      (nsub_at2 +13)	/ 'VIEWPORT'	/
	   DATA num_qualifs      (nsub_at2 +13)	/ nqual2n	/
	   DATA qualifier_pointer(nsub_at2 +13)	/ nqual_at2n	/
* SHOW VIEW /ALL/ ****
		DATA qualifiers  (nqual_at2n+0)	/ 'ALL'		/
		DATA qualifiers  (nqual_at2n+1)	/ '****'	/
		DATA qualifiers  (nqual_at2n+2)	/ '****'	/

* SHOW TRANSFORMS
	   DATA subcommands      (nsub_at2 +14)	/ 'TRANFORM'	/
	   DATA num_qualifs      (nsub_at2 +14)	/ nqual2o	/
	   DATA qualifier_pointer(nsub_at2 +14)	/ nqual_at2o	/
* SHOW TRANS / ALL
		DATA qualifiers  (nqual_at2o+0)	/ 'ALL'		/
* SHOW ALIAS				
	   DATA subcommands      (nsub_at2 +15)	/ 'ALIAS'	/
	   DATA num_qualifs      (nsub_at2 +15)	/ nqual2p	/
	   DATA qualifier_pointer(nsub_at2 +15)	/ nqual_at2p	/
* SHOW ALIAS / ****
		DATA qualifiers  (nqual_at2p+0)	/ 'ALL'		/
* SHOW SYMBOL
	   DATA subcommands      (nsub_at2 +16)	/ 'SYMBOL'	/
	   DATA num_qualifs      (nsub_at2 +16)	/ nqual2q	/
	   DATA qualifier_pointer(nsub_at2 +16)	/ nqual_at2q	/
* SHOW SYMBOL / ALL
		DATA qualifiers  (nqual_at2q+0)	/ 'ALL'		/
* SHOW ****							  ! unused(r)
	   DATA subcommands      (nsub_at2 +17)	/ '****'	/
	   DATA num_qualifs      (nsub_at2 +17)	/ nqual2r	/
	   DATA qualifier_pointer(nsub_at2 +17)	/ nqual_at2r	/
* SHOW **** / ****
		DATA qualifiers  (nqual_at2r+0)	/ 'ALL'		/
* SHOW ****							  ! unused(s)
	   DATA subcommands      (nsub_at2 +18)	/ '****'	/
	   DATA num_qualifs      (nsub_at2 +18)	/ nqual2s	/
	   DATA qualifier_pointer(nsub_at2 +18)	/ nqual_at2s	/
* SHOW **** / ****
		DATA qualifiers  (nqual_at2s+0)	/ 'ALL'		/
* SHOW ****							  ! unused(t)
	   DATA subcommands      (nsub_at2 +19)	/ '****'	/
	   DATA num_qualifs      (nsub_at2 +19)	/ nqual2t	/
	   DATA qualifier_pointer(nsub_at2 +19)	/ nqual_at2t	/
* SHOW ****
		DATA qualifiers  (nqual_at2t+0)	/ 'ALL'		/
* SHOW FUNCTION
	   DATA subcommands      (nsub_at2 +20)	/ 'FUNCTION'	/
	   DATA num_qualifs      (nsub_at2 +20)	/ nqual2u	/
	   DATA qualifier_pointer(nsub_at2 +20)	/ nqual_at2u	/
* SHOW FUNCTION / ALL / BRIEF / EXTERNAL / INTERNAL
		DATA qualifiers  (nqual_at2u+0)	/ 'ALL'		/
		DATA qualifiers  (nqual_at2u+1)	/ 'BRIEF'	/
		DATA qualifiers  (nqual_at2u+2)	/ 'EXTERNAL'	/
		DATA qualifiers  (nqual_at2u+3)	/ 'INTERNAL'	/
		DATA qualifiers  (nqual_at2u+4)	/ 'DETAILS'	/
		DATA qualifiers  (nqual_at2u+5)	/ '****'	/
		DATA qualifiers  (nqual_at2u+6)	/ '****'	/
* SHOW QUERIES							 
	   DATA subcommands      (nsub_at2 +21)	/ 'QUERIES'	/
	   DATA num_qualifs      (nsub_at2 +21)	/ nqual2v	/
	   DATA qualifier_pointer(nsub_at2 +21)	/ nqual_at2v	/
* SHOW QUERIES / ****
		DATA qualifiers  (nqual_at2v+0)	/ 'ALL'		/
		DATA qualifiers  (nqual_at2v+1)	/ '****'	/
		DATA qualifiers  (nqual_at2v+2)	/ '****'	/

* CANCEL	 - command # 3
	DATA commands(3)			/ 'CANCEL'	/
	DATA num_subcommands(3)			/ nsub3		/
	DATA subcommand_pointer(3)		/ nsub_at3	/
* CANCEL '    '
	   DATA subcommands      (nsub_at3 + 0)	/ '    '	/
	   DATA num_qualifs      (nsub_at3 + 0)	/ nqual3a	/
	   DATA qualifier_pointer(nsub_at3 + 0)	/ nqual_at3a	/
* CANCEL WINDOW
	   DATA subcommands      (nsub_at3 + 1)	/ 'WIND'	/
	   DATA num_qualifs      (nsub_at3 + 1)	/ nqual3b	/
	   DATA qualifier_pointer(nsub_at3 + 1)	/ nqual_at3b	/
* CANCEL WINDOW /ALL
		DATA qualifiers  (nqual_at3b+0)	/ 'ALL '	/
* CANCEL REGION
	   DATA subcommands      (nsub_at3 + 2)	/ 'REGION'	/
	   DATA num_qualifs      (nsub_at3 + 2)	/ nqual3c	/
	   DATA qualifier_pointer(nsub_at3 + 2)	/ nqual_at3c	/
* CANCEL REGIONS /ALL/I/J/K/L/X/Y/Z/T
		DATA qualifiers  (nqual_at3c+0)	/ 'ALL '	/
		DATA qualifiers  (nqual_at3c+1)	/ 'I   '	/
		DATA qualifiers  (nqual_at3c+2)	/ 'J   '	/
		DATA qualifiers  (nqual_at3c+3)	/ 'K   '	/
		DATA qualifiers  (nqual_at3c+4)	/ 'L   '	/
		DATA qualifiers  (nqual_at3c+5)	/ 'X   '	/
		DATA qualifiers  (nqual_at3c+6)	/ 'Y   '	/
		DATA qualifiers  (nqual_at3c+7)	/ 'Z   '	/
		DATA qualifiers  (nqual_at3c+8)	/ 'T   '	/
* CANCEL MEMORY
	   DATA subcommands      (nsub_at3 + 3)	/ 'MEMORY'	/
	   DATA num_qualifs      (nsub_at3 + 3)	/ nqual3d	/
	   DATA qualifier_pointer(nsub_at3 + 3)	/ nqual_at3d	/
* CANCEL MEMORY /ALL
		DATA qualifiers  (nqual_at3d+0)	/ 'ALL '	/
		DATA qualifiers  (nqual_at3d+1)	/ 'TEMPORY'	/
		DATA qualifiers  (nqual_at3d+2)	/ 'PERMANT'	/
* CANCEL EXPRSIONION
	   DATA subcommands      (nsub_at3 + 4)	/ 'EXPRSION'	/
	   DATA num_qualifs      (nsub_at3 + 4)	/ nqual3e	/
	   DATA qualifier_pointer(nsub_at3 + 4)	/ nqual_at3e	/
* CANCEL EXPRSION /ALL
		DATA qualifiers  (nqual_at3e+0)	/ 'ALL '	/
* CANCEL LIST
	   DATA subcommands      (nsub_at3 + 5)	/ 'LIST'	/
	   DATA num_qualifs      (nsub_at3 + 5)	/ nqual3f	/
	   DATA qualifier_pointer(nsub_at3 + 5)	/ nqual_at3f	/
* CANCEL LIST /ALL, /PRECISION, /FILE, /FORMAT
		DATA qualifiers  (nqual_at3f+0)	/ 'ALL '	/
		DATA qualifiers  (nqual_at3f+1)	/ 'PRECSION'	/
		DATA qualifiers  (nqual_at3f+2)	/ 'FILE'	/
		DATA qualifiers  (nqual_at3f+3)	/ 'FORMAT'	/
		DATA qualifiers  (nqual_at3f+4)	/ 'HEADING'	/
		DATA qualifiers  (nqual_at3f+5)	/ 'APPEND'	/
* CANCEL DATA_SET
	   DATA subcommands      (nsub_at3 + 6)	/ 'DATA'	/
	   DATA num_qualifs      (nsub_at3 + 6)	/ nqual3g	/
	   DATA qualifier_pointer(nsub_at3 + 6)	/ nqual_at3g	/
* CANCEL DATA /ALL
		DATA qualifiers  (nqual_at3g+0)	/ 'ALL '	/
		DATA qualifiers  (nqual_at3g+1)	/ 'NOERROR'	/
* CANCEL MODE
	   DATA subcommands      (nsub_at3 + 7)	/ 'MODE'	/
	   DATA num_qualifs      (nsub_at3 + 7)	/ nqual3h	/
	   DATA qualifier_pointer(nsub_at3 + 7)	/ nqual_at3h	/
* CANCEL MODE / ****
		DATA qualifiers  (nqual_at3h+0)	/ '****'	/
* CANCEL MOVIE
	   DATA subcommands      (nsub_at3 + 8)	/ 'MOVIE'	/
	   DATA num_qualifs      (nsub_at3 + 8)	/ nqual3i	/
	   DATA qualifier_pointer(nsub_at3 + 8)	/ nqual_at3i	/
* CANCEL MOVIE /ALL
		DATA qualifiers  (nqual_at3i+0)	/ 'ALL '	/
* CANCEL VIEWPORT
	   DATA subcommands      (nsub_at3 + 9)	/ 'VIEWPORT'	/
	   DATA num_qualifs      (nsub_at3 + 9)	/ nqual3j	/
	   DATA qualifier_pointer(nsub_at3 + 9)	/ nqual_at3j	/
		DATA qualifiers  (nqual_at3j+0)	/ '****' /
* CANCEL VARIABLE
	   DATA subcommands      (nsub_at3 +10)	/ 'VARIABLE'	/
	   DATA num_qualifs      (nsub_at3 +10)	/ nqual3k	/
	   DATA qualifier_pointer(nsub_at3 +10)	/ nqual_at3k	/
		DATA qualifiers  (nqual_at3k+0)	/ 'ALL ' /
		DATA qualifiers  (nqual_at3k+1)	/ 'DATASET' /
		DATA qualifiers  (nqual_at3k+2)	/ '****' /
* CANCEL AXIS							
	   DATA subcommands      (nsub_at3 +11)	/ 'AXIS'	/
	   DATA num_qualifs      (nsub_at3 +11)	/ nqual3l	/
	   DATA qualifier_pointer(nsub_at3 +11)	/ nqual_at3l	/
		DATA qualifiers  (nqual_at3l+0)	/ 'MODULO'	/
* CANCEL GRID
	   DATA subcommands      (nsub_at3 +12)	/ 'GRID'	/
	   DATA num_qualifs      (nsub_at3 +12)	/ nqual3m	/
	   DATA qualifier_pointer(nsub_at3 +12)	/ nqual_at3m	/
		DATA qualifiers  (nqual_at3m+0)	/ '****'	/
* CANCEL ****							  ! unused(n)
	   DATA subcommands      (nsub_at3 +13)	/ '****'	/
	   DATA num_qualifs      (nsub_at3 +13)	/ nqual3n	/
	   DATA qualifier_pointer(nsub_at3 +13)	/ nqual_at3n	/
		DATA qualifiers  (nqual_at3n+0)	/ '****'	/
* CANCEL ****							  ! unused(o)
	   DATA subcommands      (nsub_at3 +14)	/ '****'	/
	   DATA num_qualifs      (nsub_at3 +14)	/ nqual3o	/
	   DATA qualifier_pointer(nsub_at3 +14)	/ nqual_at3o	/
		DATA qualifiers  (nqual_at3o+0)	/ '****'	/
* CANCEL ALIAS
	   DATA subcommands      (nsub_at3 +15)	/ 'ALIAS'	/
	   DATA num_qualifs      (nsub_at3 +15)	/ nqual3p	/
	   DATA qualifier_pointer(nsub_at3 +15)	/ nqual_at3p	/
		DATA qualifiers  (nqual_at3p+0)	/ 'ALL'	/
* CANCEL SYMBOL							  
	   DATA subcommands      (nsub_at3 +16)	/ 'SYMBOL'	/
	   DATA num_qualifs      (nsub_at3 +16)	/ nqual3q	/
	   DATA qualifier_pointer(nsub_at3 +16)	/ nqual_at3q	/
		DATA qualifiers  (nqual_at3q+0)	/ 'ALL'	/
* CANCEL ****							  ! unused(r)
	   DATA subcommands      (nsub_at3 +17)	/ '****'	/
	   DATA num_qualifs      (nsub_at3 +17)	/ nqual3r	/
	   DATA qualifier_pointer(nsub_at3 +17)	/ nqual_at3r	/
		DATA qualifiers  (nqual_at3r+0)	/ '****'	/
* CANCEL ****							  ! unused(s)
	   DATA subcommands      (nsub_at3 +18)	/ '****'	/
	   DATA num_qualifs      (nsub_at3 +18)	/ nqual3s	/
	   DATA qualifier_pointer(nsub_at3 +18)	/ nqual_at3s	/
		DATA qualifiers  (nqual_at3s+0)	/ '****'	/
		DATA qualifiers  (nqual_at3s+1)	/ '****'	/
		DATA qualifiers  (nqual_at3s+2)	/ '****'	/
* CANCEL ****							  ! unused(t)
	   DATA subcommands      (nsub_at3 +19)	/ '****'	/
	   DATA num_qualifs      (nsub_at3 +19)	/ nqual3t	/
	   DATA qualifier_pointer(nsub_at3 +19)	/ nqual_at3t	/
		DATA qualifiers  (nqual_at3t+0)	/ '****'	/
		DATA qualifiers  (nqual_at3t+1)	/ '****'	/
		DATA qualifiers  (nqual_at3t+2)	/ '****'	/

* CONTOUR - command # 4
	DATA commands(4)			/ 'CONTOUR'	/
	DATA num_subcommands(4)			/ nsub4		/
	DATA subcommand_pointer(4)		/ nsub_at4	/
* CONTOUR '    '
	   DATA subcommands      (nsub_at4 + 0)	/ '    '	/
	   DATA num_qualifs      (nsub_at4 + 0)	/ nqual4a	/
	   DATA qualifier_pointer(nsub_at4 + 0)	/ nqual_at4a	/
* CONTOUR/I/J/K/L/X/Y/Z/T/D/OVERLAY/SET_UP/FRAME/TRANSPOSE
		DATA qualifiers  (nqual_at4a+0)	/ 'I'	/
		DATA qualifiers  (nqual_at4a+1)	/ 'J'	/
		DATA qualifiers  (nqual_at4a+2)	/ 'K'	/
		DATA qualifiers  (nqual_at4a+3)	/ 'L'	/
		DATA qualifiers  (nqual_at4a+4)	/ 'X'	/
		DATA qualifiers  (nqual_at4a+5)	/ 'Y'	/
		DATA qualifiers  (nqual_at4a+6)	/ 'Z'	/
		DATA qualifiers  (nqual_at4a+7)	/ 'T'	/
		DATA qualifiers  (nqual_at4a+8)	/ 'OVERLAY'/	!/overlays
		DATA qualifiers  (nqual_at4a+9)	/ 'SET_UP'/	!/set_up
		DATA qualifiers  (nqual_at4a+10)/ 'FRAME'/	!/frame
		DATA qualifiers  (nqual_at4a+11)/ 'D'   /	!/data_set
		DATA qualifiers  (nqual_at4a+12)/ 'TRANPOSE'/	!/transpose
		DATA qualifiers  (nqual_at4a+13)/ 'FILL'/	!
		DATA qualifiers  (nqual_at4a+14)/ 'LINE'/	!
		DATA qualifiers  (nqual_at4a+15)/ 'NOLABEL'/	!/nolabel
		DATA qualifiers  (nqual_at4a+16)/ 'LEVELS'/	!/levels_same
		DATA qualifiers  (nqual_at4a+17)/ 'KEY' /	!
		DATA qualifiers  (nqual_at4a+18)/ 'NOKEY'/	!
		DATA qualifiers  (nqual_at4a+19)/ 'PALETTE'/	!
		DATA qualifiers  (nqual_at4a+20)/ 'XLIMITS'/	!/xlimits
		DATA qualifiers  (nqual_at4a+21)/ 'YLIMITS'/	!/ylimits
		DATA qualifiers  (nqual_at4a+22)/ 'TITLE'/	!/title
		DATA qualifiers  (nqual_at4a+23)/ 'COLOR'/	!/color
		DATA qualifiers  (nqual_at4a+24)/ 'NOAXES'/	!/noaxes
		DATA qualifiers  (nqual_at4a+25)/ 'PATTERN'/	!/pattern=
		DATA qualifiers  (nqual_at4a+26)/ 'SIZE'/	!/size
		DATA qualifiers  (nqual_at4a+27)/ 'SPACING'/	!/spacing
		DATA qualifiers  (nqual_at4a+28)/ 'SIGDIG'/	!/sigdig
		DATA qualifiers  (nqual_at4a+29)/ 'PEN'/	!/pen (denig!!)
		DATA qualifiers  (nqual_at4a+30)/ 'HLIMITS'/	! horiz
		DATA qualifiers  (nqual_at4a+31)/ 'VLIMITS'/	! vertical
		DATA qualifiers  (nqual_at4a+32)/ '****'/	!
		DATA qualifiers  (nqual_at4a+33)/ '****'/	!


* LIST	 - command # 5
	DATA commands(5)			/ 'LIST'	/
	DATA num_subcommands(5)			/ nsub5		/
	DATA subcommand_pointer(5)		/ nsub_at5	/
* LIST '    '
	   DATA subcommands      (nsub_at5 + 0)	/ '    '	/
	   DATA num_qualifs      (nsub_at5 + 0)	/ nqual5a	/
	   DATA qualifier_pointer(nsub_at5 + 0)	/ nqual_at5a	/
* LIST/I/J/K/L/X/Y/Z/T/D /HEAD /NOHEAD /G /FILE /APPEND /ORDER /FORMAT /QUIET
		DATA qualifiers  (nqual_at5a+0)	/ 'I'	/
		DATA qualifiers  (nqual_at5a+1)	/ 'J'	/
		DATA qualifiers  (nqual_at5a+2)	/ 'K'	/
		DATA qualifiers  (nqual_at5a+3)	/ 'L'	/
		DATA qualifiers  (nqual_at5a+4)	/ 'X'	/
		DATA qualifiers  (nqual_at5a+5)	/ 'Y'	/
		DATA qualifiers  (nqual_at5a+6)	/ 'Z'	/
		DATA qualifiers  (nqual_at5a+7)	/ 'T'	/
		DATA qualifiers  (nqual_at5a+8)	/ 'D'	/
		DATA qualifiers  (nqual_at5a+9)	/ 'HEADING'/
		DATA qualifiers  (nqual_at5a+10)/ 'NOHEAD'/
		DATA qualifiers  (nqual_at5a+11)/ 'SINGLE'/
		DATA qualifiers  (nqual_at5a+12)/ 'FILE'/
		DATA qualifiers  (nqual_at5a+13)/ 'APPEND'/
		DATA qualifiers  (nqual_at5a+14)/ 'ORDER'/
		DATA qualifiers  (nqual_at5a+15)/ 'FORMAT'/
		DATA qualifiers  (nqual_at5a+16)/ 'TITLE'/
		DATA qualifiers  (nqual_at5a+17)/ 'PRECISN'/ !used to be EPKEY 
		DATA qualifiers  (nqual_at5a+18)/ 'RIGID'/ ! /RIGID (CDF)
		DATA qualifiers  (nqual_at5a+19)/ 'ILIMITS'/ 
		DATA qualifiers  (nqual_at5a+20)/ 'JLIMITS'/
		DATA qualifiers  (nqual_at5a+21)/ 'KLIMITS'/
		DATA qualifiers  (nqual_at5a+22)/ 'LLIMITS'/
		DATA qualifiers  (nqual_at5a+23)/ 'XLIMITS'/
		DATA qualifiers  (nqual_at5a+24)/ 'YLIMITS'/
		DATA qualifiers  (nqual_at5a+25)/ 'ZLIMITS'/
		DATA qualifiers  (nqual_at5a+26)/ 'TLIMITS'/
		DATA qualifiers  (nqual_at5a+27)/ 'CLOBBER'/
		DATA qualifiers  (nqual_at5a+28)/ 'QUIET'/
		DATA qualifiers  (nqual_at5a+29)/ '****'/

* PLOT - command # 6
	DATA commands(6)			/ 'PLOT'	/
	DATA num_subcommands(6)			/ nsub6		/
	DATA subcommand_pointer(6)		/ nsub_at6	/
* PLOT '    '
	   DATA subcommands      (nsub_at6 + 0)	/ '    '	/
	   DATA num_qualifs      (nsub_at6 + 0)	/ nqual6a	/
	   DATA qualifier_pointer(nsub_at6 + 0)	/ nqual_at6a	/
* PLOT/I/J/K/L/X/Y/Z/T/D/OVERLAY/SET_UP/FRAME/TRANSPOSE/VS
		DATA qualifiers  (nqual_at6a+0)	/ 'I'	/
		DATA qualifiers  (nqual_at6a+1)	/ 'J'	/
		DATA qualifiers  (nqual_at6a+2)	/ 'K'	/
		DATA qualifiers  (nqual_at6a+3)	/ 'L'	/
		DATA qualifiers  (nqual_at6a+4)	/ 'X'	/
		DATA qualifiers  (nqual_at6a+5)	/ 'Y'	/
		DATA qualifiers  (nqual_at6a+6)	/ 'Z'	/
		DATA qualifiers  (nqual_at6a+7)	/ 'T'	/
		DATA qualifiers  (nqual_at6a+8)	/ 'OVERLAY'/	!/overlays
		DATA qualifiers  (nqual_at6a+9)	/ 'SET_UP'/	!/set_up
		DATA qualifiers  (nqual_at6a+10)/ 'FRAME'/	!/frame
		DATA qualifiers  (nqual_at6a+11)/ 'D'   /	!/data_set
		DATA qualifiers  (nqual_at6a+12)/ 'TRANPOSE'/	!/transpose
		DATA qualifiers  (nqual_at6a+13)/ 'VS  '/	!/versus
		DATA qualifiers  (nqual_at6a+14)/ 'SYMBOL'/	!/symbol
		DATA qualifiers  (nqual_at6a+15)/ 'NOLABEL'/	!/nolabel
		DATA qualifiers  (nqual_at6a+16)/ 'LINE'/	!/line
		DATA qualifiers  (nqual_at6a+17)/ 'COLOR'/	!/color
		DATA qualifiers  (nqual_at6a+18)/ 'THICKNESS'/	!/thickness
		DATA qualifiers  (nqual_at6a+19)/ 'SIZE'/	!/size
		DATA qualifiers  (nqual_at6a+20)/ 'XLIMITS'/	!/xlimits
		DATA qualifiers  (nqual_at6a+21)/ 'YLIMITS'/	!/ylimits
		DATA qualifiers  (nqual_at6a+22)/ 'TITLE'/	!/title
		DATA qualifiers  (nqual_at6a+23)/ '****'/	!/unused
		DATA qualifiers  (nqual_at6a+24)/ 'NOAXES'/	!/noaxes
		DATA qualifiers  (nqual_at6a+25)/ '****'/	!/unused
		DATA qualifiers  (nqual_at6a+26)/ '****'/	!/unused
		DATA qualifiers  (nqual_at6a+27)/ '****'/	!/unused
		DATA qualifiers  (nqual_at6a+28)/ '****'/	!/unused
		DATA qualifiers  (nqual_at6a+29)/ '****'/	!/unused
		DATA qualifiers  (nqual_at6a+30)/ 'HLIMITS'/	! horiz
		DATA qualifiers  (nqual_at6a+31)/ 'VLIMITS'/	! vertical
		DATA qualifiers  (nqual_at6a+32)/ '****'/	!/unused
		DATA qualifiers  (nqual_at6a+33)/ '****'/	!/unused

* GO	 - command # 7
	DATA commands(7)			/ 'GO  '	/
	DATA num_subcommands(7)			/ nsub7		/
	DATA subcommand_pointer(7)		/ nsub_at7	/
* GO '    '
	   DATA subcommands      (nsub_at7 + 0)	/ '    '	/
	   DATA num_qualifs      (nsub_at7 + 0)	/ nqual7a	/
	   DATA qualifier_pointer(nsub_at7 + 0)	/ nqual_at7a	/
* GO (8 unused qualifiers)
		DATA qualifiers  (nqual_at7a+0)	/ 'HELP'	/
		DATA qualifiers  (nqual_at7a+1)	/ '****'	/
		DATA qualifiers  (nqual_at7a+2)	/ '****'	/
		DATA qualifiers  (nqual_at7a+3)	/ '****'	/
		DATA qualifiers  (nqual_at7a+4)	/ '****'	/
		DATA qualifiers  (nqual_at7a+5)	/ '****'	/
		DATA qualifiers  (nqual_at7a+6)	/ '****'	/
		DATA qualifiers  (nqual_at7a+7)	/ '****'	/
		DATA qualifiers  (nqual_at7a+8)	/ '****'	/

* HELP	 - command # 8
	DATA commands(8)			/ 'HELP'	/
	DATA num_subcommands(8)			/ nsub8		/
	DATA subcommand_pointer(8)		/ nsub_at8	/
* HELP '    '
	   DATA subcommands      (nsub_at8 + 0)	/ '    '	/
	   DATA num_qualifs      (nsub_at8 + 0)	/ nqual8a	/
	   DATA qualifier_pointer(nsub_at8 + 0)	/ nqual_at8a	/

* LOAD - command # 9
	DATA commands(9)			/ 'LOAD'	/
	DATA num_subcommands(9)			/ nsub9		/
	DATA subcommand_pointer(9)		/ nsub_at9	/
* LOAD '    '
	   DATA subcommands      (nsub_at9 + 0)	/ '    '	/
	   DATA num_qualifs      (nsub_at9 + 0)	/ nqual9a	/
	   DATA qualifier_pointer(nsub_at9 + 0)	/ nqual_at9a	/
* LOAD /TEMPORARY /PERMANENT /I/J/K/L/X/Y/Z/T/D /NAME
		DATA qualifiers  (nqual_at9a+0)	/ 'TEMPORY'/
		DATA qualifiers  (nqual_at9a+1)	/ 'PERMANT'/
		DATA qualifiers  (nqual_at9a+2)	/ 'I'	/
		DATA qualifiers  (nqual_at9a+3)	/ 'J'	/
		DATA qualifiers  (nqual_at9a+4)	/ 'K'	/
		DATA qualifiers  (nqual_at9a+5)	/ 'L'	/
		DATA qualifiers  (nqual_at9a+6)	/ 'X'	/
		DATA qualifiers  (nqual_at9a+7)	/ 'Y'	/
		DATA qualifiers  (nqual_at9a+8)	/ 'Z'	/
		DATA qualifiers  (nqual_at9a+9)	/ 'T'	/
		DATA qualifiers  (nqual_at9a+10)/ 'D'	/
		DATA qualifiers  (nqual_at9a+11)/ 'NAME'/
		DATA qualifiers  (nqual_at9a+12)/ '****'/
		DATA qualifiers  (nqual_at9a+13)/ '****'/
		DATA qualifiers  (nqual_at9a+14)/ '****'/

* DEFINE - command # 10
	DATA commands(10)			/ 'DEFINE'	/
	DATA num_subcommands(10)		/ nsub10	/
	DATA subcommand_pointer(10)		/ nsub_at10	/
* DEFINE '    '
	   DATA subcommands      (nsub_at10 + 0)/ '    '	/
	   DATA num_qualifs      (nsub_at10 + 0)/ nqual10a	/
	   DATA qualifier_pointer(nsub_at10 + 0)/ nqual_at10a	/
* DEFINE REGION
	   DATA subcommands      (nsub_at10 + 1)/ 'REGION'	/
	   DATA num_qualifs      (nsub_at10 + 1)/ nqual10b	/
	   DATA qualifier_pointer(nsub_at10 + 1)/ nqual_at10b	/
* DEFINE REGION /I/J/K/L/X/Y/Z/T /DEFAULT / ****
		DATA qualifiers  (nqual_at10b+0)/ 'I'	/
		DATA qualifiers  (nqual_at10b+1)/ 'J'	/
		DATA qualifiers  (nqual_at10b+2)/ 'K'	/
		DATA qualifiers  (nqual_at10b+3)/ 'L'	/
		DATA qualifiers  (nqual_at10b+4)/ 'X'	/
		DATA qualifiers  (nqual_at10b+5)/ 'Y'	/
		DATA qualifiers  (nqual_at10b+6)/ 'Z'	/
		DATA qualifiers  (nqual_at10b+7)/ 'T'	/
		DATA qualifiers  (nqual_at10b+8)/ 'DEFAULT'/
		DATA qualifiers  (nqual_at10b+9)/ '****'/
		DATA qualifiers  (nqual_at10b+10)/ 'DX'/
		DATA qualifiers  (nqual_at10b+11)/ 'DY'/
		DATA qualifiers  (nqual_at10b+12)/ 'DZ'/
		DATA qualifiers  (nqual_at10b+13)/ 'DT'/
		DATA qualifiers  (nqual_at10b+14)/ 'DI'/
		DATA qualifiers  (nqual_at10b+15)/ 'DJ'/
		DATA qualifiers  (nqual_at10b+16)/ 'DK'/
		DATA qualifiers  (nqual_at10b+17)/ 'DL'/
* DEFINE GRID
	   DATA subcommands      (nsub_at10 + 2)/ 'GRID'	/
	   DATA num_qualifs      (nsub_at10 + 2)/ nqual10c	/
	   DATA qualifier_pointer(nsub_at10 + 2)/ nqual_at10c	/
* DEFINE GRID /X/Y/Z/T /FILE /LIKE
		DATA qualifiers  (nqual_at10c+ 0)/ 'X'	  /
		DATA qualifiers  (nqual_at10c+ 1)/ 'Y'	  /
		DATA qualifiers  (nqual_at10c+ 2)/ 'Z'	  /
		DATA qualifiers  (nqual_at10c+ 3)/ 'T'	  /
		DATA qualifiers  (nqual_at10c+ 4)/ 'FILE' /
		DATA qualifiers  (nqual_at10c+ 5)/ 'LIKE' /
		DATA qualifiers  (nqual_at10c+ 6)/ '****' /
		DATA qualifiers  (nqual_at10c+ 7)/ '****' /
		DATA qualifiers  (nqual_at10c+ 8)/ '****' /
		DATA qualifiers  (nqual_at10c+ 9)/ '****' /
		DATA qualifiers  (nqual_at10c+10)/ '****' /
* DEFINE VARIABLE
	   DATA subcommands      (nsub_at10 + 3)/ 'VARIABLE'	/
	   DATA num_qualifs      (nsub_at10 + 3)/ nqual10d	/
	   DATA qualifier_pointer(nsub_at10 + 3)/ nqual_at10d	/
* DEFINE VARIABLE /TITLE
		DATA qualifiers  (nqual_at10d+ 0)/ 'TITLE' /
		DATA qualifiers  (nqual_at10d+ 1)/ 'UNITS' /
		DATA qualifiers  (nqual_at10d+ 2)/ 'QUIET' /
		DATA qualifiers  (nqual_at10d+ 3)/ 'DATASET' /
		DATA qualifiers  (nqual_at10d+ 4)/ 'BAD' /
		DATA qualifiers  (nqual_at10d+ 5)/ '****' /
* DEFINE AXIS
	   DATA subcommands      (nsub_at10 + 4)/ 'AXIS'	/
	   DATA num_qualifs      (nsub_at10 + 4)/ nqual10e	/
	   DATA qualifier_pointer(nsub_at10 + 4)/ nqual_at10e	/
* DEFINE AXIS /X/Y/Z/T /FILE /UNITS ...
		DATA qualifiers  (nqual_at10e+ 0)/ 'X'	  /
		DATA qualifiers  (nqual_at10e+ 1)/ 'Y'	  /
		DATA qualifiers  (nqual_at10e+ 2)/ 'Z'	  /
		DATA qualifiers  (nqual_at10e+ 3)/ 'T'	  /
		DATA qualifiers  (nqual_at10e+ 4)/ 'FILE' /
		DATA qualifiers  (nqual_at10e+ 5)/ 'UNITS' /
		DATA qualifiers  (nqual_at10e+ 6)/ 'T0' /
		DATA qualifiers  (nqual_at10e+ 7)/ 'NAME' /
		DATA qualifiers  (nqual_at10e+ 8)/ 'FROMDATA' /
		DATA qualifiers  (nqual_at10e+ 9)/ 'DEPTH' /
		DATA qualifiers  (nqual_at10e+ 10)/ 'MODULO' /
		DATA qualifiers  (nqual_at10e+ 11)/ 'NPOINTS' /
		DATA qualifiers  (nqual_at10e+ 12)/ 'EDGES' /
		DATA qualifiers  (nqual_at10e+ 13)/ '****' /
		DATA qualifiers  (nqual_at10e+ 14)/ '****' /
* DEFINE VIEW
	   DATA subcommands      (nsub_at10 + 5)/ 'VIEWPORT'	/
	   DATA num_qualifs      (nsub_at10 + 5)/ nqual10f	/
	   DATA qualifier_pointer(nsub_at10 + 5)/ nqual_at10f	/
* DEFINE VIEW /SIZE/ORIGIN/CLIP/ ****
		DATA qualifiers  (nqual_at10f+ 0)/ 'TEXT' /
		DATA qualifiers  (nqual_at10f+ 1)/ 'XLIMITS' /
		DATA qualifiers  (nqual_at10f+ 2)/ 'YLIMITS' /
		DATA qualifiers  (nqual_at10f+ 3)/ 'SIZE' /
		DATA qualifiers  (nqual_at10f+ 4)/ 'ORIGIN' /
		DATA qualifiers  (nqual_at10f+ 5)/ 'CLIP' /
		DATA qualifiers  (nqual_at10f+ 6)/ '****' /
		DATA qualifiers  (nqual_at10f+ 7)/ '****' /
* DEFINE ALIAS
	   DATA subcommands      (nsub_at10 + 6)/ 'ALIAS'	/
	   DATA num_qualifs      (nsub_at10 + 6)/ nqual10g	/
	   DATA qualifier_pointer(nsub_at10 + 6)/ nqual_at10g	/
* DEFINE ALIAS / ****
		DATA qualifiers  (nqual_at10g+ 0)/ '****' /
		DATA qualifiers  (nqual_at10g+ 1)/ '****' /
* DEFINE SYMBOL
	   DATA subcommands      (nsub_at10 + 7)/ 'SYMBOL'	/
	   DATA num_qualifs      (nsub_at10 + 7)/ nqual10h	/
	   DATA qualifier_pointer(nsub_at10 + 7)/ nqual_at10h	/
* DEFINE SYMBOL / ****
		DATA qualifiers  (nqual_at10h+ 0)/ '****' /
		DATA qualifiers  (nqual_at10h+ 1)/ '****' /

* EXIT	 - command # 11
	DATA commands(11)			/ 'EXIT'	/
	DATA num_subcommands(11)		/ nsub11	/
	DATA subcommand_pointer(11)		/ nsub_at11	/
* EXIT '    '
	   DATA subcommands      (nsub_at11 + 0)/ '    '	/
	   DATA num_qualifs      (nsub_at11 + 0)/ nqual11a	/
	   DATA qualifier_pointer(nsub_at11 + 0)/ nqual_at11a	/
* EXIT/COMMAND_FILE
		DATA qualifiers  (nqual_at11a+0)/ 'COMMAND'	/

* MESSAGE	 - command # 12
	DATA commands(12)			/ 'MESSAGE'	/
	DATA num_subcommands(12)		/ nsub12	/
	DATA subcommand_pointer(12)		/ nsub_at12	/
* MESSAGE '    '
	   DATA subcommands      (nsub_at12 + 0)/ '    '	/
	   DATA num_qualifs      (nsub_at12 + 0)/ nqual12a	/
	   DATA qualifier_pointer(nsub_at12 + 0)/ nqual_at12a	/
* MESSAGE/CONTINUE
		DATA qualifiers  (nqual_at12a+0)/ 'CONTINUE'	/
* MESSAGE/QUIET
		DATA qualifiers  (nqual_at12a+1)/ 'QUIET'	/
		DATA qualifiers  (nqual_at12a+2)/ 'JOURNAL'	/
		DATA qualifiers  (nqual_at12a+3)/ 'ERROR'	/

* VECTOR - command # 13
	DATA commands(13)			/ 'VECTOR'	/
	DATA num_subcommands(13)		/ nsub13	/
	DATA subcommand_pointer(13)		/ nsub_at13	/
* VECTOR '    '
	   DATA subcommands      (nsub_at13 + 0)/ '    '	/
	   DATA num_qualifs      (nsub_at13 + 0)/ nqual13a	/
	   DATA qualifier_pointer(nsub_at13 + 0)/ nqual_at13a	/
* VECTOR/I/J/K/L/X/Y/Z/T/D/OVERLAY/SET_UP/FRAME/TRANSPOSE
		DATA qualifiers  (nqual_at13a+0)/ 'I'	/
		DATA qualifiers  (nqual_at13a+1)/ 'J'	/
		DATA qualifiers  (nqual_at13a+2)/ 'K'	/
		DATA qualifiers  (nqual_at13a+3)/ 'L'	/
		DATA qualifiers  (nqual_at13a+4)/ 'X'	/
		DATA qualifiers  (nqual_at13a+5)/ 'Y'	/
		DATA qualifiers  (nqual_at13a+6)/ 'Z'	/
		DATA qualifiers  (nqual_at13a+7)/ 'T'	/
		DATA qualifiers  (nqual_at13a+8)/ 'OVERLAY'/	!/overlays
		DATA qualifiers  (nqual_at13a+9)/ 'SET_UP'/	!/set_up
		DATA qualifiers  (nqual_at13a+10)/ 'FRAME'/	!/frame
		DATA qualifiers  (nqual_at13a+11)/ 'D'   /	!/data_set
		DATA qualifiers  (nqual_at13a+12)/ 'TRANPOSE'/	!/transpose
		DATA qualifiers  (nqual_at13a+13)/ '****'/	!
		DATA qualifiers  (nqual_at13a+14)/ 'ASPECT'/	!/aspect
		DATA qualifiers  (nqual_at13a+15)/ 'NOLABEL'/	!/nolabel
		DATA qualifiers  (nqual_at13a+16)/ 'LENGTH'/	!/length_same
		DATA qualifiers  (nqual_at13a+17)/ 'XSKIP'/	!/xskip
		DATA qualifiers  (nqual_at13a+18)/ 'YSKIP'/	!/yskip
		DATA qualifiers  (nqual_at13a+19)/ 'THICK'/	!/thick
		DATA qualifiers  (nqual_at13a+20)/ 'XLIMITS'/	!/xlimits
		DATA qualifiers  (nqual_at13a+21)/ 'YLIMITS'/	!/ylimits
		DATA qualifiers  (nqual_at13a+22)/ 'TITLE'/	!/title
		DATA qualifiers  (nqual_at13a+23)/ 'COLOR'/	!/color
		DATA qualifiers  (nqual_at13a+24)/ 'NOAXES'/	!/noaxes
		DATA qualifiers  (nqual_at13a+25)/ 'STREAMLI'/  !/streamline
		DATA qualifiers  (nqual_at13a+26)/ 'DENSITY'/	!/density=
		DATA qualifiers  (nqual_at13a+27)/ '****'/	!
		DATA qualifiers  (nqual_at13a+28)/ '****'/	!
		DATA qualifiers  (nqual_at13a+29)/ 'PEN'/	! pen (denigr!)
		DATA qualifiers  (nqual_at13a+30)/ 'HLIMITS'/	! horiz
		DATA qualifiers  (nqual_at13a+31)/ 'VLIMITS'/	! vertical
		DATA qualifiers  (nqual_at13a+32)/ '****'/	!
		DATA qualifiers  (nqual_at13a+33)/ '****'/	!

* PPLUS	 - command # 14
	DATA commands(14)			/ 'PPLUS'	/
	DATA num_subcommands(14)		/ nsub14	/
	DATA subcommand_pointer(14)		/ nsub_at14	/
* PPLUS '    '
	   DATA subcommands      (nsub_at14 + 0)/ '    '	/
	   DATA num_qualifs      (nsub_at14 + 0)/ nqual14a	/
	   DATA qualifier_pointer(nsub_at14 + 0)/ nqual_at14a	/
* PPLUS/RESET
		DATA qualifiers  (nqual_at14a+0)/ 'RESET' /
		DATA qualifiers  (nqual_at14a+1)/ '****' /

* FRAME	 - command # 15
	DATA commands(15)			/ 'FRAME'	/
	DATA num_subcommands(15)		/ nsub15	/
	DATA subcommand_pointer(15)		/ nsub_at15	/
* FRAME '    '
	   DATA subcommands      (nsub_at15 + 0)/ '    '	/
	   DATA num_qualifs      (nsub_at15 + 0)/ nqual15a	/
	   DATA qualifier_pointer(nsub_at15 + 0)/ nqual_at15a	/
* FRAME/FORMAT=GIF_or_HDF/FILE=filename/ **** / ****
		DATA qualifiers  (nqual_at15a+0)/ 'FORMAT'	/
		DATA qualifiers  (nqual_at15a+1)/ 'FILE'	/
		DATA qualifiers  (nqual_at15a+2)/ '****'	/
		DATA qualifiers  (nqual_at15a+3)/ '****'	/

* REPEAT - command # 16
	DATA commands(16)			/ 'REPEAT'	/
	DATA num_subcommands(16)		/ nsub16	/
	DATA subcommand_pointer(16)		/ nsub_at16	/
* REPEAT '    '
	   DATA subcommands      (nsub_at16 + 0)/ '    '	/
	   DATA num_qualifs      (nsub_at16 + 0)/ nqual16a	/
	   DATA qualifier_pointer(nsub_at16 + 0)/ nqual_at16a	/
* REPEAT/I/J/K/L/X/Y/Z/T
		DATA qualifiers  (nqual_at16a+0)/ 'I'	/
		DATA qualifiers  (nqual_at16a+1)/ 'J'	/
		DATA qualifiers  (nqual_at16a+2)/ 'K'	/
		DATA qualifiers  (nqual_at16a+3)/ 'L'	/
		DATA qualifiers  (nqual_at16a+4)/ 'X'	/
		DATA qualifiers  (nqual_at16a+5)/ 'Y'	/
		DATA qualifiers  (nqual_at16a+6)/ 'Z'	/
		DATA qualifiers  (nqual_at16a+7)/ 'T'	/
		DATA qualifiers  (nqual_at16a+8)/ '*****' /   ! future "QUIET"
		DATA qualifiers  (nqual_at16a+9)/ '****'  /

* STATISTICS - command # 17
	DATA commands(17)			/ 'STAT'	/
	DATA num_subcommands(17)		/ nsub17	/
	DATA subcommand_pointer(17)		/ nsub_at17	/
* STAT '    '
	   DATA subcommands      (nsub_at17 + 0)	/ '    '	/
	   DATA num_qualifs      (nsub_at17 + 0)	/ nqual17a	/
	   DATA qualifier_pointer(nsub_at17 + 0)	/ nqual_at17a	/
* STAT /I/J/K/L/X/Y/Z/T/D/G
		DATA qualifiers  (nqual_at17a+0)	/ 'BRIEF'/
		DATA qualifiers  (nqual_at17a+1)	/ '****'/
		DATA qualifiers  (nqual_at17a+2)	/ 'I'	/
		DATA qualifiers  (nqual_at17a+3)	/ 'J'	/
		DATA qualifiers  (nqual_at17a+4)	/ 'K'	/
		DATA qualifiers  (nqual_at17a+5)	/ 'L'	/
		DATA qualifiers  (nqual_at17a+6)	/ 'X'	/
		DATA qualifiers  (nqual_at17a+7)	/ 'Y'	/
		DATA qualifiers  (nqual_at17a+8)	/ 'Z'	/
		DATA qualifiers  (nqual_at17a+9)	/ 'T'	/
		DATA qualifiers  (nqual_at17a+10)	/ 'D'	/
		DATA qualifiers  (nqual_at17a+11)	/ '****'/
		DATA qualifiers  (nqual_at17a+12)	/ '****'/
		DATA qualifiers  (nqual_at17a+13)	/ '****'/
		DATA qualifiers  (nqual_at17a+14)	/ '****'/
		DATA qualifiers  (nqual_at17a+15)	/ '****'/

* SHADE - command # 18
	DATA commands(18)			/ 'SHADE'	/
	DATA num_subcommands(18)		/ nsub18	/
	DATA subcommand_pointer(18)		/ nsub_at18	/
* SHADE '    '
	   DATA subcommands      (nsub_at18 + 0)	/ '    '	/
	   DATA num_qualifs      (nsub_at18 + 0)	/ nqual18a	/
	   DATA qualifier_pointer(nsub_at18 + 0)	/ nqual_at18a	/
* SHADE/I/J/K/L/X/Y/Z/T/D/OVERLAY/SET_UP/FRAME/TRANSPOSE
		DATA qualifiers  (nqual_at18a+0)/ 'I'	/
		DATA qualifiers  (nqual_at18a+1)/ 'J'	/
		DATA qualifiers  (nqual_at18a+2)/ 'K'	/
		DATA qualifiers  (nqual_at18a+3)/ 'L'	/
		DATA qualifiers  (nqual_at18a+4)/ 'X'	/
		DATA qualifiers  (nqual_at18a+5)/ 'Y'	/
		DATA qualifiers  (nqual_at18a+6)/ 'Z'	/
		DATA qualifiers  (nqual_at18a+7)/ 'T'	/
		DATA qualifiers  (nqual_at18a+8)/ 'OVERLAY'/	!/overlays
		DATA qualifiers  (nqual_at18a+9)/ 'SET_UP'/	!/set_up
		DATA qualifiers  (nqual_at18a+10)/ 'FRAME'/	!/frame
		DATA qualifiers  (nqual_at18a+11)/ 'D'   /	!/data_set
		DATA qualifiers  (nqual_at18a+12)/ 'TRANPOSE'/	!/transpose
		DATA qualifiers  (nqual_at18a+13)/ '****'/	!/ ****
		DATA qualifiers  (nqual_at18a+14)/ 'LINE'/	!/lines
		DATA qualifiers  (nqual_at18a+15)/ 'NOLABEL'/	!/nolabel
		DATA qualifiers  (nqual_at18a+16)/ 'LEVELS'/	!/levels_same
		DATA qualifiers  (nqual_at18a+17)/ 'KEY' /	!/key
		DATA qualifiers  (nqual_at18a+18)/ 'NOKEY'/	!/nokey
		DATA qualifiers  (nqual_at18a+19)/ 'PALETTE'/	!/palette=
		DATA qualifiers  (nqual_at18a+20)/ 'XLIMITS'/	!/xlimits
		DATA qualifiers  (nqual_at18a+21)/ 'YLIMITS'/	!/ylimits
		DATA qualifiers  (nqual_at18a+22)/ 'TITLE'/	!/title
		DATA qualifiers  (nqual_at18a+23)/ '****'/	! 
		DATA qualifiers  (nqual_at18a+24)/ 'NOAXES'/	!/noaxes
		DATA qualifiers  (nqual_at18a+25)/ 'PATTERN'/   !/pattern=
		DATA qualifiers  (nqual_at18a+26)/ '****'/	!
		DATA qualifiers  (nqual_at18a+27)/ '****'/	!
		DATA qualifiers  (nqual_at18a+28)/ '****'/	!
		DATA qualifiers  (nqual_at18a+29)/ '****'/	!
		DATA qualifiers  (nqual_at18a+30)/ 'HLIMITS'/	! horiz
		DATA qualifiers  (nqual_at18a+31)/ 'VLIMITS'/	! vertical
		DATA qualifiers  (nqual_at18a+32)/ '****'/	!
		DATA qualifiers  (nqual_at18a+33)/ '****'/	!

* SPAWN - command # 19
	DATA commands(19)			/ 'SPAWN'	/
	DATA num_subcommands(19)		/ nsub19	/
	DATA subcommand_pointer(19)		/ nsub_at19	/
* SPAWN '    '
	   DATA subcommands      (nsub_at19 + 0)	/ '    '	/
	   DATA num_qualifs      (nsub_at19 + 0)	/ nqual19a	/
	   DATA qualifier_pointer(nsub_at19 + 0)	/ nqual_at19a	/

* USER - command # 20
	DATA commands(20)			/ 'USER'	/
	DATA num_subcommands(20)		/ nsub20	/
	DATA subcommand_pointer(20)		/ nsub_at20	/
* USER '    '
	   DATA subcommands      (nsub_at20 + 0)	/ '    '	/
	   DATA num_qualifs      (nsub_at20 + 0)	/ nqual20a	/
	   DATA qualifier_pointer(nsub_at20 + 0)	/ nqual_at20a	/
* USER/OPT1/OPT2
		DATA qualifiers  (nqual_at20a+0)/ 'OPT1'	/
		DATA qualifiers  (nqual_at20a+1)/ 'OPT2'	/
		DATA qualifiers  (nqual_at20a+2)/ 'COMMAND'	/
		DATA qualifiers  (nqual_at20a+3)/ 'I'		/
		DATA qualifiers  (nqual_at20a+4)/ 'J'		/
		DATA qualifiers  (nqual_at20a+5)/ 'K'		/
		DATA qualifiers  (nqual_at20a+6)/ 'L'		/
		DATA qualifiers  (nqual_at20a+7)/ 'X'		/
		DATA qualifiers  (nqual_at20a+8)/ 'Y'		/
		DATA qualifiers  (nqual_at20a+9)/ 'Z'		/
		DATA qualifiers  (nqual_at20a+10)/ 'T'		/
		DATA qualifiers  (nqual_at20a+11)/ 'D'		/
		DATA qualifiers  (nqual_at20a+12)/ 'FILE'	/
		DATA qualifiers  (nqual_at20a+13)/ 'FORMAT'	/

* WIRE - command # 21
	DATA commands(21)			/ 'WIRE'	/
	DATA num_subcommands(21)		/ nsub21	/
	DATA subcommand_pointer(21)		/ nsub_at21	/
* WIRE '    '
	   DATA subcommands      (nsub_at21 + 0)	/ '    '	/
	   DATA num_qualifs      (nsub_at21 + 0)	/ nqual21a	/
	   DATA qualifier_pointer(nsub_at21 + 0)	/ nqual_at21a	/
* WIRE/I/J/K/L/X/Y/Z/T/D/OVERLAY/SET_UP/FRAME/TRANSPOSE
		DATA qualifiers  (nqual_at21a+0) / 'I'	/
		DATA qualifiers  (nqual_at21a+1) / 'J'	/
		DATA qualifiers  (nqual_at21a+2) / 'K'	/
		DATA qualifiers  (nqual_at21a+3) / 'L'	/
		DATA qualifiers  (nqual_at21a+4) / 'X'	/
		DATA qualifiers  (nqual_at21a+5) / 'Y'	/
		DATA qualifiers  (nqual_at21a+6) / 'Z'	/
		DATA qualifiers  (nqual_at21a+7) / 'T'	/
		DATA qualifiers  (nqual_at21a+8) / 'OVERLAY'/ 
		DATA qualifiers  (nqual_at21a+9) / 'SET_UP'/  
		DATA qualifiers  (nqual_at21a+10)/ 'FRAME'/	
		DATA qualifiers  (nqual_at21a+11)/ 'D'   /	
		DATA qualifiers  (nqual_at21a+12)/ 'VIEWPOIN'/	
		DATA qualifiers  (nqual_at21a+13)/ 'ZLIMITS'/	
		DATA qualifiers  (nqual_at21a+14)/ 'TRANPOSE'/  
		DATA qualifiers  (nqual_at21a+15)/ 'NOLABEL'/   
		DATA qualifiers  (nqual_at21a+16)/ 'ZSCALE'/ 
		DATA qualifiers  (nqual_at21a+17)/ '****'/
		DATA qualifiers  (nqual_at21a+18)/ '****'/
		DATA qualifiers  (nqual_at21a+19)/ '****'/
		DATA qualifiers  (nqual_at21a+20)/ '****'/
		DATA qualifiers  (nqual_at21a+21)/ '****'/
		DATA qualifiers  (nqual_at21a+22)/ 'TITLE'/
		DATA qualifiers  (nqual_at21a+23)/ '****'/
		DATA qualifiers  (nqual_at21a+24)/ '****'/

* QUERY - command # 22
	DATA commands(22)			/ 'QUERY'	/
	DATA num_subcommands(22)		/ nsub22	/
	DATA subcommand_pointer(22)		/ nsub_at22	/
* QUERY '    '
	   DATA subcommands      (nsub_at22 + 0)	/ '    '	/
	   DATA num_qualifs      (nsub_at22 + 0)	/ nqual22a	/
	   DATA qualifier_pointer(nsub_at22 + 0)	/ nqual_at22a	/
* QUERY /FILE
		DATA qualifiers  (nqual_at22a+0)/ 'ALL'	/
		DATA qualifiers  (nqual_at22a+1)/ 'FILE'	/
		DATA qualifiers  (nqual_at22a+2)/ 'IGNORE'	/

* IF - command # 23
	DATA commands(23)			/ 'IF  '	/
	DATA num_subcommands(23)		/ nsub23	/
	DATA subcommand_pointer(23)		/ nsub_at23	/
* IF '    '
	   DATA subcommands      (nsub_at23 + 0)	/ '    '	/
	   DATA num_qualifs      (nsub_at23 + 0)	/ nqual23a	/
	   DATA qualifier_pointer(nsub_at23 + 0)	/ nqual_at23a	/
* IF / ****
		DATA qualifiers  (nqual_at23a+0)/ '****'	/
		DATA qualifiers  (nqual_at23a+1)/ '****'	/

* ELSE - command # 24
	DATA commands(24)			/ 'ELSE'	/
	DATA num_subcommands(24)		/ nsub24	/
	DATA subcommand_pointer(24)		/ nsub_at24	/
* ELSE '    '
	   DATA subcommands      (nsub_at24 + 0)	/ '    '	/
	   DATA num_qualifs      (nsub_at24 + 0)	/ nqual24a	/
	   DATA qualifier_pointer(nsub_at24 + 0)	/ nqual_at24a	/
* ELSE / ****
		DATA qualifiers  (nqual_at24a+0)/ '****'	/
		DATA qualifiers  (nqual_at24a+1)/ '****'	/

* ELIF - command # 25
	DATA commands(25)			/ 'ELIF'	/
	DATA num_subcommands(25)		/ nsub25	/
	DATA subcommand_pointer(25)		/ nsub_at25	/
* ELIF '    '
	   DATA subcommands      (nsub_at25 + 0)	/ '    '	/
	   DATA num_qualifs      (nsub_at25 + 0)	/ nqual25a	/
	   DATA qualifier_pointer(nsub_at25 + 0)	/ nqual_at25a	/
* ELIF / ****
		DATA qualifiers  (nqual_at25a+0)/ '****'	/
		DATA qualifiers  (nqual_at25a+1)/ '****'	/

* ENDIF - command # 26
	DATA commands(26)			/ 'ENDIF'	/
	DATA num_subcommands(26)		/ nsub26	/
	DATA subcommand_pointer(26)		/ nsub_at26	/
* ENDIF '    '
	   DATA subcommands      (nsub_at26 + 0)	/ '    '	/
	   DATA num_qualifs      (nsub_at26 + 0)	/ nqual26a	/
	   DATA qualifier_pointer(nsub_at26 + 0)	/ nqual_at26a	/
* ENDIF / ****
		DATA qualifiers  (nqual_at26a+0)/ '****'	/
		DATA qualifiers  (nqual_at26a+1)/ '****'	/

* POLYGON - command # 27
	DATA commands(27)			/ 'POLYGON'	/
	DATA num_subcommands(27)			/ nsub6		/
	DATA subcommand_pointer(27)		/ nsub_at27	/
* POLYGON '    '
	   DATA subcommands      (nsub_at27 + 0)/ '    '	/
	   DATA num_qualifs      (nsub_at27 + 0)/ nqual27a	/
	   DATA qualifier_pointer(nsub_at27 + 0)/ nqual_at27a	/
* POLYGON/I/J/K/L/X/Y/Z/T/D/OVERLAY/SET_UP/FRAME/TRANSPOSE/COORD_AX
		DATA qualifiers  (nqual_at27a+0)/ 'I'	/
		DATA qualifiers  (nqual_at27a+1)/ 'J'	/
		DATA qualifiers  (nqual_at27a+2)/ 'K'	/
		DATA qualifiers  (nqual_at27a+3)/ 'L'	/
		DATA qualifiers  (nqual_at27a+4)/ 'X'	/
		DATA qualifiers  (nqual_at27a+5)/ 'Y'	/
		DATA qualifiers  (nqual_at27a+6)/ 'Z'	/
		DATA qualifiers  (nqual_at27a+7)/ 'T'	/
		DATA qualifiers  (nqual_at27a+8)/ 'OVERLAY'/	!/overlays
		DATA qualifiers  (nqual_at27a+9)/ 'SET_UP'/	!/set_up
		DATA qualifiers  (nqual_at27a+10)/ 'FRAME'/	!/frame
		DATA qualifiers  (nqual_at27a+11)/ 'D'   /	!/data_set
		DATA qualifiers  (nqual_at27a+12)/ 'TRANPOSE'/	!/transpose
		DATA qualifiers  (nqual_at27a+13)/ 'COORD_AX'/	!/coord_ax
		DATA qualifiers  (nqual_at27a+14)/ '****'/	!/unused
		DATA qualifiers  (nqual_at27a+15)/ 'NOLABEL'/	!/nolabel
		DATA qualifiers  (nqual_at27a+16)/ 'LEVELS'/	!/levels_same
		DATA qualifiers  (nqual_at27a+17)/ 'LINE'/	!/line
		DATA qualifiers  (nqual_at27a+18)/ 'COLOR'/	!/color
		DATA qualifiers  (nqual_at27a+19)/ 'PALETTE'/	!/palette=
		DATA qualifiers  (nqual_at27a+20)/ 'XLIMITS'/	!/xlimits
		DATA qualifiers  (nqual_at27a+21)/ 'YLIMITS'/	!/ylimits
		DATA qualifiers  (nqual_at27a+22)/ 'TITLE'/	!/title
		DATA qualifiers  (nqual_at27a+23)/ 'THICKNES'/	!/thickness
		DATA qualifiers  (nqual_at27a+24)/ 'NOAXES'/	!/noaxes
	        DATA qualifiers  (nqual_at27a+25)/ 'PATTERN'/   !/pattern=
		DATA qualifiers  (nqual_at27a+26)/ 'FILL'/	!/fill
		DATA qualifiers  (nqual_at27a+27)/ 'KEY'/	!/key
		DATA qualifiers  (nqual_at27a+28)/ 'NOKEY'/	!/nokey
		DATA qualifiers  (nqual_at27a+29)/ '****'/	!unused
		DATA qualifiers  (nqual_at27a+30)/ 'HLIMITS'/	! horizontal
		DATA qualifiers  (nqual_at27a+31)/ 'VLIMITS'/	! vertical
		DATA qualifiers  (nqual_at27a+32)/ '****'/	!unused
		DATA qualifiers  (nqual_at27a+33)/ '****'/	!unused

* **** - command # 28
	DATA commands(28)			/ '****'	/
	DATA num_subcommands(28)		/ nsub28	/
	DATA subcommand_pointer(28)		/ nsub_at28	/
* **** '   
	   DATA subcommands      (nsub_at28 + 0)	/ '    '	/
	   DATA num_qualifs      (nsub_at28 + 0)	/ nqual28a	/
	   DATA qualifier_pointer(nsub_at28 + 0)	/ nqual_at28a	/
* **** / ****
		DATA qualifiers  (nqual_at28a+0)/ '****'	/
		DATA qualifiers  (nqual_at28a+1)/ '****'	/

* **** - command # 29
	DATA commands(29)			/ '****'	/
	DATA num_subcommands(29)		/ nsub29	/
	DATA subcommand_pointer(29)		/ nsub_at29	/
* **** '    '
	   DATA subcommands      (nsub_at29 + 0)	/ '    '	/
	   DATA num_qualifs      (nsub_at29 + 0)	/ nqual29a	/
	   DATA qualifier_pointer(nsub_at29 + 0)	/ nqual_at29a	/
* **** / ****
		DATA qualifiers  (nqual_at29a+0)/ '****'	/
		DATA qualifiers  (nqual_at29a+1)/ '****'	/

* **** - command # 30
	DATA commands(30)			/ '****'	/
	DATA num_subcommands(30)		/ nsub30	/
	DATA subcommand_pointer(30)		/ nsub_at30	/
* **** '    '
	   DATA subcommands      (nsub_at30 + 0)	/ '    '	/
	   DATA num_qualifs      (nsub_at30 + 0)	/ nqual30a	/
	   DATA qualifier_pointer(nsub_at30 + 0)	/ nqual_at30a	/
* **** / ****
		DATA qualifiers  (nqual_at30a+0)/ '****'	/
		DATA qualifiers  (nqual_at30a+1)/ '****'	/

* ******* ALIASES *********
	DATA	alias_name(1)	/ 'LET'/,
     .		alias_len(1)	/ 3 /,
     .		alias(1)	/ 'DEFINE VARIABLE'/,
     .		alias_name(2)	/ 'FILE'/,
     .		alias_len(2)	/ 4 /,
     .		alias(2)	/ 'SET DATA/EZ'/,
     .		alias_name(3)	/ 'QUIT' /,  ! or "q" for happy Unix users
     .		alias_len(3)	/ 4 /,
     .		alias(3)	/ 'EXIT' /,
     .		alias_name(4)	/ 'REGION' /,
     .		alias_len(4)	/ 4 /,
     .		alias(4)	/ 'SET REGION' /,
     .		alias_name(5)	/ 'SAY' /,
     .		alias_len(5)	/ 16 /,
     .		alias(5)	/ 'MESSAGE/CONTINUE' /
     .		alias_name(6)	/ 'FILL' /,
     .		alias_len(6)	/ 12 /,
     .		alias(6)	/ 'CONTOUR/FILL' /
     .		alias_name(7)	/ 'ALIAS' /,
     .		alias_len(7)	/ 12 /,
     .		alias(7)	/ 'DEFINE ALIAS' /
     .		alias_name(8)	/ 'UNALIAS' /,
     .		alias_len(8)	/ 12 /,
     .		alias(8)	/ 'CANCEL ALIAS' /
     .		alias_name(9)	/ 'USE' /,
     .		alias_len(9)	/ 16 /,
     .		alias(9)	/ 'SET DAT/FORM=CDF' /
     .		alias_name(10)	/ 'SAVE' /,
     .		alias_len(10)	/ 15 /,
     .		alias(10)	/ 'LIST/FORMAT=CDF' /,
     .		alias_name(11)	/ 'PALETTE' /,
     .		alias_len(11)	/ 20 /,
     .		alias(11)	/ 'PPL SHASET SPECTRUM=' /,
     .		alias_name(12)	/ 'LABEL' /,
     .		alias_len(12)	/ 10 /,
     .		alias(12)	/ 'PPL %LABEL' /,
     .		alias_name(13)	/ 'ANIMATE' /,
     .		alias_len(13)	/ 10 /,
     .		alias(13)	/ 'SPAWN xds&' /,
     .		alias_name(14)	/ 'PATTERN' /,
     .		alias_len(14)	/ 19 /,
     .		alias(14)	/ 'PPL PATSET PATTERN=' /
     .		alias_name(15)	/ 'PAUSE' /,
     .		alias_len(15)	/ 7 /,
     .		alias(15)	/ 'MESSAGE' /,
     .		alias_name(16)	/ 'WHERE' /,
     .		alias_len(16)	/ 10 /,
     .		alias(16)	/ 'PPL %WHERE' /
     .		alias_name(17)	/ 'STREAMLI' /,
     .		alias_len(17)	/ 17 /,
     .		alias(17)	/ 'VECTOR/STREAMLINE' /
        DATA
     .		(alias_name(i),i=num_predefined_alias+1,150)
     .                   / 133*unspecified_name4 /,
     .		(alias_len(i),i=num_predefined_alias+1,150)/ 133*4 /,
     .		(alias(i),i=num_predefined_alias+1,150)
     .                   / 133*unspecified_name4 /


	DATA queries(1) / 'STATUS' /,
     .	     queries(2) / 'MESSAGE' /,
     .	     queries(3) / 'DSET' /,
     .	     queries(4) / 'VARIABLE' /,
     .	     queries(5) / 'GRID' /,
     .	     queries(6) / 'AXIS' /,
     .	     queries(7) / 'COORDS' /,
     .	     queries(8) / 'TRANS' /,
     .	     queries(9) / 'TRNARG' /,
     .	     queries(10) / 'LVARS' /,
     .	     queries(11) / 'DVARS' /,
     .	     queries(12) / 'VBACK' /,
     .	     queries(13) / 'DBACK' /,
     .	     queries(14) / 'WINDOWS' /,
     .	     queries(15) / 'WCURRENT' /,
     .	     queries(16) / 'VPORTS' /,
     .	     queries(17) / 'VCURRENT' /,
     .	     queries(18) / 'VIEW' /,
     .	     queries(19) / 'DCURRENT' /,
     .	     queries(20) / 'SPAN' /,
     .	     queries(21) / 'GAXIS' /,
     .	     queries(22) / 'GCOORDS' /,
     .	     queries(23) / 'SSPOINT' /,
     .	     queries(24) / 'WWPOINT' /,
     .	     queries(25) / ' ' /
	DATA (queries(i),i=26,max_queries) / 25*unspecified_name4 /
	DATA num_queries /0/   ! unused

	END









