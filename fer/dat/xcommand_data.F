	BLOCK DATA XCOMMAND_DATA

*  This software was developed by the Thermal Modeling and Analysis
*  Project(TMAP) of the National Oceanographic and Atmospheric
*  Administration''s (NOAA) Pacific Marine Environmental Lab(PMEL),
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
*
* !!!!!!!!!!
* Use caution when defining new commands:
* 1- Ferret commands are matched to the first 4 characters only. Make sure
*    new commands are unique to the first 4 characters.
* 2- If you are tempted to reorder the lists of commands or subcommands, know 
*    that users areprone to even more abbreviation. E.g. "can v" is used for 
*    "cancel viewport" all over the place, and that happens ONLY because "VIEWPORT" 
*    comes before "VARIABLE" in the subcommands for CANCEL.
* !!!!!!!!!!
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
*	     *sh* - added DEFINE AXIS/EDGES and SHOW AXIS/X/Y/Z/T/E/F/IJ/K/L 
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
*      10/00 *acm* - added VECTOR/FLOWLINE (alias FLOWLINE)
*      1 /01 *sh* - added PLOT/STEP[=connected]
*       1/01 *acm* - added DEFINE AXIS/CALENDAR qualifier
*       5/01 *acm* - added PLOT/DASH[=(dn1,up1,dn2,up2) qualifier
*       1/01 *acm* - added /CALENDAR qualifier
* V531: 5/01 *kob* - replace old ANIMATE alias which spawned a call
*	             to what must be the now-defunct xds with new 
*		     animate alias which uses internal double buffering
* V533: 6/01 *sh* - added SET VARIABLE/NAME=
*            *acm* - added PLOT/VLOG, PLOT/HLOG: plot qualifiers to set log,
*                    inverse log axes and log depth axes.
*       7/01 *acm* - added /AXES[=TOP,BOTTOM,LEFT,RIGHT]  for 
*		     PLOT, CONTOUR, POLYGON, SHADE, VECTOR.
*       8/01 *kob* - add LOOP, PROMPT, SCRIPT, PROGRAM qualifiers 
*                    to EXIT command
* V540: 9/01 *sh* - added DEFINE VIEWPORT/AXES
*	10/01 *sh* - added alias COLUMNS for SET DATA/FORMAT=Delimited
*	11/01 *sh* - added list/width=
* V541: 4/02 *acm* - remove CONTOUR/HLOG/VLOG; mistakenly put in here!!
*                    note they were not used by xeq_contour...
* V541: 5/02 *acm* - add VECTOR/NOKEY
*       6/02 *acm* - add CANCEL AXIS/ALL
* V552: 4/02 *acm* - add SET VARIABLE/SCALEFAC/OFFSET
* V553: 7/03 *acm* - add SHOW DATA/XML for DAS, DDS output
* V553: 7/03 *acm* - add SHOW AXIS/XML 
* V553: 9/03 *acm* - add SHOW AXIS/DODS for binary output of axis coordinates
*                       also needs /FILE=  and /APPEND
* V554: 10/03 *acm* - add SHOW AXIS/EDGES/DODS for binary output of axis coordinates
* V554:  1/04 *acm* - add REPEAT/RANGE=/NAME= qualifiers
* V554:  2/04 *acm* - add /HGRAT=/VGRAT qualifiers to plot commands, setting 
*                     GRATICULE lines at tic marks
* V570:  5/04 *acm* - Add DEFINE AXIS/BOUNDS 
*                     and LIST/EDGES, LIST/BOUNDS (for NetCDF output only)
*       *acm* 6/05 - add SET AXIS/CALENDAR /T0 /UNITS
*       *acm* 6/05 - add CANCEL AXIS/DEPTH
* V580: 10/04 *acm* - add SHOW VARIABLES/XML
* V580: 11/04 *acm* - add PLOT/NOKEY
* V581:  2/05 *acm* - SHADE/MODULO for curvilinear plots
* V581:  4/05 *acm* - CONTOUR/MODULO for curvilinear plots
* V581:  5/05 *acm* - add EXIT/CYCLE
* V590:  8/05 *acm* - add LIST/NOROWLAB

* V600:  8/05 *acm* - add LIST/NOROWLAB
* V600:  9/05 *acm* - add SET WINDOW/TITLE
* V600:  3/07 *acm* - fixing bug 439&1390. Add logical variable its_action_command 
* V600   6/05 *acm* - SHOW ATTRIBUTE[/ALL][/DATA=] var 
*                   - SHOW DATA/ATTR output all attributes for nc files
* V600:  8/05 *acm* - add LIST/NOROWLAB
*       *acm* 8/05 - add DEFINE ATTRIBUTE
*       *acm* 9/05 - add SET ATTRIBUTE
*       *acm*10/05 - add CANCEL ATTRIBUTE
* V600: 2/06 *acm* - add SET VARIABLE/OUTTYPE for attribute-handling additions
*       7/06 *jli* - add SET AXIS/STRIDE /OFFSET for grid striding
*       7/06 *jli* - add CANCEL AXIS/STRIDES
*       *acm* 7/06 -     CANCEL ATTRIBUTE /OUTPUT /DATA
* V603: 5/07 *acm* - Fix bug found by Andy Jacobson, in line 1465 index should be 30 not 29.
* V606  8/07 *acm* - DEFINE AXIS/QUIET
* V6.2  *acm* 5/08 new qualifier FRAME/TRANSPARENT
* V6.2  *acm* 2/09 new qualifier LIST/KEEP_AXISNAMES
* V6.2  *acm* 5/09 new alias KEYMARK
* V6.6  *acm* 1/10 new qualifiers for LIST netCDF4: NCFORMAT,
*                  XCHUNK,YCHUNK,ZCHUNK,TCHUNK,DEFLATE,SHUFFLE,ENDIAN
* V6.5  *acm* 3/10 new qualifiers on (some) SHOW commands and on MESSAGE: 
*                  /OUTFILE= /APPEND /CLOBBER
* V6.6  *acm* 2/10 new qualifiers for SET LIST netCDF4: NCFORMAT,
*                  XCHUNK,YCHUNK,ZCHUNK,TCHUNK,DEFLATE,SHUFFLE,ENDIAN
*       *kms* 5/10 add SET AXIS/REGULAR to force interpreting the axis as regular
*       *kms* 5/10 add SET REDIRECT /TEE /JOURNAL /FILE /APPEND /CLOBBER 
*                  and CANCEL REDIRECT for stdout/stderr redirection
*       *kms* 7/10 add SET DATA /BROWSE to browse for the dataset
* V6.63 *acm* 7/10 Add VECTOR/KEY to force key even if /NOLAB
*       *kms* 7/10 add SET DATA /BROWSE to browse for the dataset
* V664  8.10 *kms* Add EXIT/TOPYTHON to exit the ferret_dispatch loop 
*                  but just re-enter the loop if not under pyferret
*                  Add DEFINE PYFUNC /NAME=<alias> python.module.name
* V68 *acm* 1/12   Add USE/STRICT; see ticket 1910.
* PyFr  *kms* 3/11 Add CANCEL PYVAR /ALL /DATASET
* PyFr  *kms* 1/12 Add SET GRAPHICS /ENGINE /ANTIALIA /NOANTIAL
* V68 *acm* 5/12   Add DEFINE DATA /AGGREGATE /NAME= /QUIET
* V683*acm* 8/12   New SET LIST/OUTTYPE, CANCEL LIST/OUTTYPE and SAVE/OUTTYPE 
* V683*acm 10/12   NEW SET AXIS/OUTTYPE to change default coordinate axis output type
* V684 *acm* 12/12 Changes for POLY/MODULO and  VECTOR/MODULO on curvilinear plots.
* V685 *acm*  1/13 DEFINE VARIABLE/REMOTE
* V685 *acm*  6/13 new qualifier SAVE/NOBOUNDS
* V685 *acm*  6/13 Dont add SAVE/NOCOORDS
* PyFr  *kms* 1/13 Remove SET GRAPHICS /ENGINE /ANTIALIA /NOANTIAL;
*                  Add /QUALITY= /AAlias /NoAAlias /ENGINE= to SET WINDOW
* PyFr  *kms* 3/13 Change /AAlias and /NoAAlias to /AntiAlia[s] and /NoAntiAl[ias]
* V685 *acm*  4/13 Add /HIDE on DEF DATA/AGG and SHOW DATA/HIDDEN
* PyFr  *kms* 6/13  Add /COLOR= /OPACITY= to specify background color
* PyFr  *kms* 7/13  Add /THICKEN to SET WINDOW to specify line thickness scaling factor
* PyFr  *kms*  7/13 Add /OPACITY to CONTOUR, PLOT, POLYGON, SHADE, VECTOR
* PyFr  *kms*  7/13 Add /XPIXELS, /YPIXELS, /XINCHES, /YINCHES to SET WINDOW and FRAME
* PyFr  *kms*  8/13 Add /ANNOTATE to FRAME
* V685 *sh*  9/13  Add SHOW VARIABLE/TREE and alias VTREE
* V685 *acm* 11/13 Add /QUIET for DEFINE ATTRIBUTE
* V690 *sh* 12/13 Add /CURVILINEAR and /LAYERZ to action commands (formerly /SIGMA)
*		  Add SET,CANCEL,SHOW VARIABLE/LAYERZ/CURVI  	  (formerly /SIGMA)
* V687  3/14 *acm* add SET AXIS/NAME
* V693+ 11/14 *sh* renaming 'sigma' as 'layerz' throughout
* V695 *acm*  2/15 Add DEFINE AGGREGATE/F alias, FMRC
* V695 *acm*  3/15 Changes for PyFerret fonts. Add SHOW TEXT, 
*             4/15 Add SET TEXT/FONT/COLOR/ITALIC/BOLD/ISIZ
*                  Add CANCEL TEXT/FONT/COLOR/ITALIC/BOLD/ISIZ
*                  No args on SHOW TEXT
* V695 *acm*  6/16 New ANNOTATE command.
* V695 *acm*  6/16 New subcommand DEFINE ANNOTATION
* v696 *acm*  6/15 PLOT/VS/GAPLOC to define seperate line segments on
*                  PLOT/VS plots (LAS ticket 1597 and others)
* v696 *acm* 11/15 SET VAR/UNITS instead of /UNIT, just for consistency
* V697 *acm* 12/15 Correct typo: LIST/SINGLY was shown as /SINGLE
* V698+ *acm* 2/16 Ticket 2362: Change /D= qualifier to /DATASET=
* V698  3/16 *acm* Removed Philander-Seigel code including diagnostic 
*                  variables, pre-defined axes and grids, cat_calc_var.
*       3/16 *sh*  added PLOT/ALONG=axis
* V698  4/16 *acm* New PLOT/SKIP= (and leave space for PyFerret PLOT/OPACITY=)
* V698  4/16 *acm* Change it to PLOT/SKIPSYM= 
* V7    *acm* 6/16 Ticket 2434: command qualifiers /DEGMINSEC= /HALFSPAC
*                  DEGMINSEC=1 for Deg-Min axis labels, =2 for Deg-Min-Sec. Default = 1
*                  /HALFSPAC adds a half space between the degree sign and E/W or N/S
* V7    6/16 *kms* Add reserved slots for PyFerret commands and qualifiers
*                  to simplify future merges of Ferret into PyFerret
* V7   10/16 *kms* New SET WINDOW /SCALE=
* V710 *acm*  3/16 Added DEFINE AXIS /LIKE=
* V710 *acm*  3/16 Ticket 2158: Add DEFINE AXIS /MONTHLY 
* V702 *sh*   1/17 Removed the USER command
*	      3/17 Added SHOW MEMORY/DIAGNOSTIC
* V720 *acm*  3/17 Ticket 2512: Add STAT/PRECISION=
* V720 *acm*  5/17 Ticket 2530 add WIRE/COLOR= 
* V722 *kms* 10/17 Remove threddsBrowser (too big, outdated, and unused)
* V723 *acm* 11/17 New SAVE/ASGLOBAL option to write global attribute to netCDF file
* V74  *acm*  2/17 New DEFINE AXIS/LIKE and DEFINE VARIABLE/LIKE
* V740 *kms*  3/18 Add space for PyFerret PLOT/HIGHLIGHT=
* V7.4+ *acm* 9/18 DSG-enabled Ferret: SET DATA/FMASK= specifies a feature mask
* V74+ *acm*  9/18 Add aliases FMASK for SET DATA/FMASK and CLEARMASK for CANCEL DATA/FMASK
* v751 *acm*  8/19 Issue 1946, allow for user-set string dimension length, SAVE/STRLEN
* v751 *acm*  8/19 New LIST/NOCOORDS to write dimensions only not coord vars. 
* v76  *acm* 11/19  Issue 1959: 
*                  SET MEMORY/SIZE= to SET MEMORY/MAX= (keep /SIZE for bkwds compatibility)
* V7.62 *acm* 8/20 DSG-enabled Ferret: SET DATA/SMASK= specifies a mask on station or
*                  trajectory for timeseriesProfile or trajectoryProfile data
*
* !!!!!!!!!!
* Use caution when defining new commands, qualifiers, or changing these data structures:
* 1- Ferret commands are matched to the first 4 characters only. Make sure
*    new commands are unique to the first 4 characters.
* 2- If you are tempted to reorder the lists of commands or subcommands, know 
*    that users often use even more abbreviation. E.g. "can v" is used for 
*    "cancel viewport" in many scripts, and that works ONLY because "VIEWPORT" 
*    comes before "VARIABLE" in the subcommands for CANCEL.
* !!!!!!!!!!
 


        IMPLICIT NONE
	include 'ferret.parm'
	include 'xcommand.cmn'

* local variable declarations:
        INTEGER i

* SET	 - command # 1
	DATA commands(1)			/ 'SET '	/
        DATA its_action_command(1)              / .FALSE.       /
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
		DATA qualifiers  (nqual_at1b+5)	/ 'TITLE'	/
		DATA qualifiers  (nqual_at1b+6)	/ 'QUALITY'	/
		DATA qualifiers  (nqual_at1b+7)	/ 'ANTIALIA'	/
		DATA qualifiers  (nqual_at1b+8)	/ 'NOANTIAL'	/
		DATA qualifiers  (nqual_at1b+9)	/ 'ENGINE'	/
		DATA qualifiers  (nqual_at1b+10)/ 'COLOR'	/
		DATA qualifiers  (nqual_at1b+11)/ 'OPACITY'	/
		DATA qualifiers  (nqual_at1b+12)/ 'THICKEN'	/
		DATA qualifiers  (nqual_at1b+13)/ 'XPIXELS'	/
		DATA qualifiers  (nqual_at1b+14)/ 'YPIXELS'	/
		DATA qualifiers  (nqual_at1b+15)/ 'XINCHES'	/
		DATA qualifiers  (nqual_at1b+16)/ 'YINCHES'	/
		DATA qualifiers  (nqual_at1b+17)/ 'TEXTPROM'	/
		DATA qualifiers  (nqual_at1b+18)/ 'OUTLINE'	/
		DATA qualifiers  (nqual_at1b+19)/ 'SCALE'	/
* SET REGION
	   DATA subcommands      (nsub_at1 + 2)	/ 'REGION'	/
	   DATA num_qualifs      (nsub_at1 + 2)	/ nqual1c	/
	   DATA qualifier_pointer(nsub_at1 + 2)	/ nqual_at1c	/
* SET REGION/I/J/K/L/M/N/X/Y/Z/T/E/F/DX/DY/DZ/DT/DE/DF/DI/DJ/DK/DL/DM/DN
		DATA qualifiers  (nqual_at1c+0)	/ 'I'	/
		DATA qualifiers  (nqual_at1c+1)	/ 'J'	/
		DATA qualifiers  (nqual_at1c+2)	/ 'K'	/
		DATA qualifiers  (nqual_at1c+3)	/ 'L'	/
		DATA qualifiers  (nqual_at1c+4)	/ 'M'	/
		DATA qualifiers  (nqual_at1c+5)	/ 'N'	/
		DATA qualifiers  (nqual_at1c+6)	/ 'X'	/
		DATA qualifiers  (nqual_at1c+7)	/ 'Y'	/
		DATA qualifiers  (nqual_at1c+8)	/ 'Z'	/
		DATA qualifiers  (nqual_at1c+9)	/ 'T'	/
		DATA qualifiers  (nqual_at1c+10)/ 'E'/
		DATA qualifiers  (nqual_at1c+11)/ 'F'/
		DATA qualifiers  (nqual_at1c+12)/ '****'/   ! need this for alignment
		DATA qualifiers  (nqual_at1c+13)/ '****'/   ! in get_delta_context
		DATA qualifiers  (nqual_at1c+14)/ 'DX'/
		DATA qualifiers  (nqual_at1c+15)/ 'DY'/
		DATA qualifiers  (nqual_at1c+16)/ 'DZ'/
		DATA qualifiers  (nqual_at1c+17)/ 'DT'/
		DATA qualifiers  (nqual_at1c+18)/ 'DE'/
		DATA qualifiers  (nqual_at1c+19)/ 'DF'/
		DATA qualifiers  (nqual_at1c+20)/ 'DI'/
		DATA qualifiers  (nqual_at1c+21)/ 'DJ'/
		DATA qualifiers  (nqual_at1c+22)/ 'DK'/
		DATA qualifiers  (nqual_at1c+23)/ 'DL'/
		DATA qualifiers  (nqual_at1c+24)/ 'DM'/
		DATA qualifiers  (nqual_at1c+25)/ 'DN'/
		DATA qualifiers  (nqual_at1c+26)/ '****'/
* SET VIEWPORT
	   DATA subcommands      (nsub_at1 + 3)	/ 'VIEWPORT'	/
	   DATA num_qualifs      (nsub_at1 + 3)	/ nqual1d	/
	   DATA qualifier_pointer(nsub_at1 + 3)	/ nqual_at1d	/
* SET VIEWPORT/SIZE/ORIGIN/CLIP
		DATA qualifiers  (nqual_at1d+0)	/ '****'	/
* SET EXPRSION
	   DATA subcommands      (nsub_at1 + 4)	/ 'EXPRSION'	/
	   DATA num_qualifs      (nsub_at1 + 4)	/ nqual1e	/
	   DATA qualifier_pointer(nsub_at1 + 4)	/ nqual_at1e	/
* SET LIST
	   DATA subcommands      (nsub_at1 + 5)	/ 'LIST'	/
	   DATA num_qualifs      (nsub_at1 + 5)	/ nqual1f	/
	   DATA qualifier_pointer(nsub_at1 + 5)	/ nqual_at1f	/
* SET LIST/PRECISION=#digits, /FILE=filename, /FORMAT=list_format, /APPEND, 
* (for netcdf4 /NCFORMAT= /ENDIAN=, /DEFLATE=, /SHUFFLE=,  /XCHUNK...
		DATA qualifiers  (nqual_at1f+0)	/ 'PRECISION'	/
		DATA qualifiers  (nqual_at1f+1)	/ 'FILE'	/
		DATA qualifiers  (nqual_at1f+2)	/ 'FORMAT'	/
		DATA qualifiers  (nqual_at1f+3)	/ 'APPEND'	/
		DATA qualifiers  (nqual_at1f+4)	/ 'HEADING'	/
		DATA qualifiers  (nqual_at1f+5)	/ 'NCFORMAT'	/
		DATA qualifiers  (nqual_at1f+6)	/ 'ENDIAN'	/
		DATA qualifiers  (nqual_at1f+7)	/ 'DEFLATE'	/
		DATA qualifiers  (nqual_at1f+8)	/ 'SHUFFLE'	/
		DATA qualifiers  (nqual_at1f+9)	/ 'XCHUNK'	/
		DATA qualifiers  (nqual_at1f+10)/ 'YCHUNK'	/
		DATA qualifiers  (nqual_at1f+11)/ 'ZCHUNK'	/
		DATA qualifiers  (nqual_at1f+12)/ 'TCHUNK'	/
		DATA qualifiers  (nqual_at1f+13)/ 'ECHUNK'	/
		DATA qualifiers  (nqual_at1f+14)/ 'FCHUNK'	/
		DATA qualifiers  (nqual_at1f+15)/ 'OUTTYPE'	/
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
		DATA qualifiers  (nqual_at1g+13) / 'DELIMITE'   /
		DATA qualifiers  (nqual_at1g+14) / 'STRICT'     /
		DATA qualifiers  (nqual_at1g+15) / 'FEATURE'    /
		DATA qualifiers  (nqual_at1g+16) / 'FMASK'      /
		DATA qualifiers  (nqual_at1g+17) / 'SMASK'       /
		DATA qualifiers  (nqual_at1g+18) / '****'       /
		DATA qualifiers  (nqual_at1g+19) / '****'       /
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
* SET VARIABLE/TITLE/UNITS/GRID/BAD/DATASET/NAME/SCALEFAC/OFFSET/OUTTYPE
		DATA qualifiers  (nqual_at1j+0)	/ 'TITLE'	/
		DATA qualifiers  (nqual_at1j+1)	/ 'UNITS'	/
		DATA qualifiers  (nqual_at1j+2)	/ 'GRID'	/
		DATA qualifiers  (nqual_at1j+3)	/ 'BAD'	/
		DATA qualifiers  (nqual_at1j+4)	/ 'DATASET'	/
		DATA qualifiers  (nqual_at1j+5)	/ 'NAME'	/
		DATA qualifiers  (nqual_at1j+6)	/ 'SCALEFAC'	/
		DATA qualifiers  (nqual_at1j+7)	/ 'OFFSET'	/
		DATA qualifiers  (nqual_at1j+8)	/ 'OUTTYPE'	/
		DATA qualifiers  (nqual_at1j+9)	/ 'LAYERZ'	/
		DATA qualifiers  (nqual_at1j+10)/ 'CURVILINEAR'	/
		DATA qualifiers  (nqual_at1j+11)/ '****'	/
		DATA qualifiers  (nqual_at1j+12)/ '****'	/
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
* SET AXIS / MODULO /DEPTH /CALENDAR /UNITS /T0 /STRIDE /OFFSET /REGULAR /OUTTYPE /NAME
		DATA qualifiers  (nqual_at1l+0)	/ 'MODULO'	/
		DATA qualifiers  (nqual_at1l+1)	/ 'DEPTH'	/
		DATA qualifiers  (nqual_at1l+2)	/ 'CALENDAR'	/
		DATA qualifiers  (nqual_at1l+3)	/ 'T0'	/
		DATA qualifiers  (nqual_at1l+4)	/ 'UNITS'	/
		DATA qualifiers  (nqual_at1l+5)	/ 'STRIDE'	/
		DATA qualifiers  (nqual_at1l+6)	/ 'OFFSET'	/
		DATA qualifiers  (nqual_at1l+7)	/ 'REGULAR'	/
		DATA qualifiers  (nqual_at1l+8)	/ 'OUTTYPE'	/
		DATA qualifiers  (nqual_at1l+9)	/ 'NAME'	/
* SET MEMORY
	   DATA subcommands      (nsub_at1 +12)	/ 'MEMORY'	/
	   DATA num_qualifs      (nsub_at1 +12)	/ nqual1m	/
	   DATA qualifier_pointer(nsub_at1 +12)	/ nqual_at1m	/
* SET MEMO / MWORDS
		DATA qualifiers  (nqual_at1m+0)	/ 'MAX'	/
		DATA qualifiers  (nqual_at1m+1)	/ 'SIZE'	/
* SET ATTRIBUTE
	   DATA subcommands      (nsub_at1 +13)	/ 'ATTRIBUT'	/
	   DATA num_qualifs      (nsub_at1 +13)	/ nqual1n	/
	   DATA qualifier_pointer(nsub_at1 +13)	/ nqual_at1n	/
* SET ATRR /TYPE /DATASET /OUTPUT /LIKE 
		DATA qualifiers  (nqual_at1n+0)	/ 'TYPE'	/
		DATA qualifiers  (nqual_at1n+1)	/ 'DATASET'	/
		DATA qualifiers  (nqual_at1n+2)	/ 'OUTPUT'	/
		DATA qualifiers  (nqual_at1n+3)	/ 'LIKE'	/
		DATA qualifiers  (nqual_at1n+4)	/ 'QUIET'	/
		DATA qualifiers  (nqual_at1n+5)	/ '****'	/
		DATA qualifiers  (nqual_at1n+6)	/ '****'	/
		DATA qualifiers  (nqual_at1n+7)	/ '****'	/
* SET NCCACHE							  ! cache chunk for reading
	   DATA subcommands      (nsub_at1 +14)	/ 'NCCACHE'	/
	   DATA num_qualifs      (nsub_at1 +14)	/ nqual1o	/
	   DATA qualifier_pointer(nsub_at1 +14)	/ nqual_at1o	/
* SET NCCACHE /SIZE/NELEMS/PREEMPT
		DATA qualifiers  (nqual_at1o+0)	/ 'SIZE'	/
		DATA qualifiers  (nqual_at1o+1)	/ 'NELEMS'	/
		DATA qualifiers  (nqual_at1o+2)	/ 'PREEMPT'	/
* SET REDIRECT							  ! redirecting stdout and/or stderr
	   DATA subcommands      (nsub_at1 +15)	/ 'REDIRECT'	/
	   DATA num_qualifs      (nsub_at1 +15)	/ nqual1p	/
	   DATA qualifier_pointer(nsub_at1 +15)	/ nqual_at1p	/
* SET REDIRECT /TEE /JOURNAL /FILE /APPEND /CLOBBER
		DATA qualifiers  (nqual_at1p+0)	/ 'TEE'		/
		DATA qualifiers  (nqual_at1p+1)	/ 'JOURNAL'	/
		DATA qualifiers  (nqual_at1p+2)	/ 'FILE'	/
		DATA qualifiers  (nqual_at1p+3)	/ 'APPEND'	/
		DATA qualifiers  (nqual_at1p+4)	/ 'CLOBBER'	/
		DATA qualifiers  (nqual_at1p+5)	/ '****'	/
		DATA qualifiers  (nqual_at1p+6)	/ '****'	/
		DATA qualifiers  (nqual_at1p+7)	/ '****'	/
* SET GIFFILE							  ! set default gif filename and the pointer used in gif mode
	   DATA subcommands      (nsub_at1 +16)	/ 'GIFFILE'	/
	   DATA num_qualifs      (nsub_at1 +16)	/ nqual1q	/
	   DATA qualifier_pointer(nsub_at1 +16)	/ nqual_at1q	/
* SET TEXT							  ! set text properties for PyFerret fonts
           DATA subcommands      (nsub_at1 +17)	/ 'TEXT'	/
           DATA num_qualifs      (nsub_at1 +17)	/ nqual1r	/
           DATA qualifier_pointer(nsub_at1 +17)	/ nqual_at1r	/
* SET TEXT /FONT /COLOR /ITALIC /BOLD 
                DATA qualifiers  (nqual_at1r+0)	/ 'FONT'	/
                DATA qualifiers  (nqual_at1r+1)	/ 'COLOR'	/
                DATA qualifiers  (nqual_at1r+2)	/ 'ITALIC'	/
                DATA qualifiers  (nqual_at1r+3)	/ 'BOLD'	/
                DATA qualifiers  (nqual_at1r+4)	/ 'ISIZ'	/
                DATA qualifiers  (nqual_at1r+5)	/ '****'	/
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
        DATA its_action_command(2)              / .FALSE.       /
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
* SHOW AXIS /ALL/I/J/K/L/M/N/X/Y/Z/T/E/F/XML/OUTFILE/APPEND/CLOBBER
		DATA qualifiers  (nqual_at2d+0)	/ 'ALL '	/
		DATA qualifiers  (nqual_at2d+1)	/ 'I'	/
		DATA qualifiers  (nqual_at2d+2)	/ 'J'	/
		DATA qualifiers  (nqual_at2d+3)	/ 'K'	/
		DATA qualifiers  (nqual_at2d+4)	/ 'L'	/
		DATA qualifiers  (nqual_at2d+5)	/ 'M'	/
		DATA qualifiers  (nqual_at2d+6)	/ 'N'	/
		DATA qualifiers  (nqual_at2d+7)	/ 'X'	/
		DATA qualifiers  (nqual_at2d+8)	/ 'Y'	/
		DATA qualifiers  (nqual_at2d+9)	/ 'Z'	/
		DATA qualifiers  (nqual_at2d+10)/ 'T'	/
		DATA qualifiers  (nqual_at2d+11)/ 'E'	/
		DATA qualifiers  (nqual_at2d+12)/ 'F'	/
		DATA qualifiers  (nqual_at2d+13)/ 'XML'/
		DATA qualifiers  (nqual_at2d+14)/ 'OUTFILE'/
		DATA qualifiers  (nqual_at2d+15)/ 'APPEND'/
		DATA qualifiers  (nqual_at2d+16)/ 'CLOBBER'/
		DATA qualifiers  (nqual_at2d+17)/ '****'/
		DATA qualifiers  (nqual_at2d+18)/ '****'/
		DATA qualifiers  (nqual_at2d+19)/ '****'/

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
* SHOW DATA_SET /ALL/BRIEF/FULL/VAR/FILE/XML/ATTR/OUTFILE/APPEND/CLOBBER/HIDDEN
		DATA qualifiers  (nqual_at2g+0)	/ 'ALL '	/
		DATA qualifiers  (nqual_at2g+1)	/ 'BRIEF'	/
		DATA qualifiers  (nqual_at2g+2)	/ 'FULL'	/
		DATA qualifiers  (nqual_at2g+3)	/ 'VARIABLE'	/
		DATA qualifiers  (nqual_at2g+4)	/ 'FILES'	/ ! deprecated
		DATA qualifiers  (nqual_at2g+5)	/ 'XML'		/
		DATA qualifiers  (nqual_at2g+6)	/ 'ATTR'	/
		DATA qualifiers  (nqual_at2g+7)	/ 'OUTFILE'	/
		DATA qualifiers  (nqual_at2g+8)	/ 'APPEND'	/
		DATA qualifiers  (nqual_at2g+9)	/ 'CLOBBER'	/
		DATA qualifiers  (nqual_at2g+10)/ 'HIDDEN'	/
		DATA qualifiers  (nqual_at2g+11)/ 'MEMBERS'	/
		DATA qualifiers  (nqual_at2g+12)/ '****'	/
		DATA qualifiers  (nqual_at2g+13)/ '****'	/
		DATA qualifiers  (nqual_at2g+14)/ '****'	/
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
* SHOW VARIABLES /ALL /DATASET  /USER /OUTFILE/APPEND/CLOBBER
* /DIAGNOSTI removed 3/2016
		DATA qualifiers  (nqual_at2j+0)	/ 'ALL '	/
		DATA qualifiers  (nqual_at2j+1)	/ 'DATASET'	/
		DATA qualifiers  (nqual_at2j+2)	/ '****'	/
		DATA qualifiers  (nqual_at2j+3)	/ 'USER'	/
		DATA qualifiers  (nqual_at2j+4)	/ 'XML'		/
		DATA qualifiers  (nqual_at2j+5)	/ 'OUTFILE'	/
		DATA qualifiers  (nqual_at2j+6)	/ 'APPEND'	/
		DATA qualifiers  (nqual_at2j+7)	/ 'CLOBBER'	/
		DATA qualifiers  (nqual_at2j+8)	/ 'TREE'	/ ! was /DEPENDENCY
		DATA qualifiers  (nqual_at2j+9)	/ 'LAYERZ'	/
		DATA qualifiers  (nqual_at2j+10)/ 'CURVILINEAR'	/
		DATA qualifiers  (nqual_at2j+11)/ '****'	/
		DATA qualifiers  (nqual_at2j+12)/ '****'	/
* SHOW COMMANDS
	   DATA subcommands      (nsub_at2 +10)	/ 'COMMANDS'	/
	   DATA num_qualifs      (nsub_at2 +10)	/ nqual2k	/
	   DATA qualifier_pointer(nsub_at2 +10)	/ nqual_at2k	/
* SHOW COMMANDS /ALL /BRIEF
		DATA qualifiers  (nqual_at2k+0)	/ 'ALL '	/
		DATA qualifiers  (nqual_at2k+1)	/ 'BRIEF'	/
* SHOW MEMORY
	   DATA subcommands      (nsub_at2 +11)	/ 'MEMORY'	/
	   DATA num_qualifs      (nsub_at2 +11)	/ nqual2l	/
	   DATA qualifier_pointer(nsub_at2 +11)	/ nqual_at2l	/
* SHOW MEMORY /ALL /TEMP /PERM /FREE
		DATA qualifiers  (nqual_at2l+0)	/ 'ALL '	/
		DATA qualifiers  (nqual_at2l+1)	/ 'DIAGNOSTIC'	/
		DATA qualifiers  (nqual_at2l+2)	/ 'TEMPORY'	/
		DATA qualifiers  (nqual_at2l+3)	/ 'PERMANT'	/
		DATA qualifiers  (nqual_at2l+4)	/ 'FREE'	/
* SHOW GRID
	   DATA subcommands      (nsub_at2 +12)	/ 'GRID'	/
	   DATA num_qualifs      (nsub_at2 +12)	/ nqual2m	/
	   DATA qualifier_pointer(nsub_at2 +12)	/ nqual_at2m	/
* SHOW GRID /ALL/I/J/K/L/M/N/X/Y/Z/T/E/F/DATASET/XML/OUTFILE/APPEND/CLOBBER 
		DATA qualifiers  (nqual_at2m+0)	/ 'ALL '	/
		DATA qualifiers  (nqual_at2m+1)	/ 'I   '	/
		DATA qualifiers  (nqual_at2m+2)	/ 'J   '	/
		DATA qualifiers  (nqual_at2m+3)	/ 'K   '	/
		DATA qualifiers  (nqual_at2m+4)	/ 'L   '	/
		DATA qualifiers  (nqual_at2m+5)	/ 'M   '	/
		DATA qualifiers  (nqual_at2m+6)	/ 'N   '	/
		DATA qualifiers  (nqual_at2m+7)	/ 'X   '	/
		DATA qualifiers  (nqual_at2m+8)	/ 'Y   '	/
		DATA qualifiers  (nqual_at2m+9)	/ 'Z   '	/
		DATA qualifiers  (nqual_at2m+10)/ 'T   '	/
		DATA qualifiers  (nqual_at2m+11)/ 'E   '	/
		DATA qualifiers  (nqual_at2m+12)/ 'F   '	/
		DATA qualifiers  (nqual_at2m+13)/ 'DYNAMIC'	/
		DATA qualifiers  (nqual_at2m+14)/ 'XML'	/
		DATA qualifiers  (nqual_at2m+15)/ 'OUTFILE'	/
		DATA qualifiers  (nqual_at2m+16)/ 'APPEND'	/
		DATA qualifiers  (nqual_at2m+17)/ 'CLOBBER'	/
		DATA qualifiers  (nqual_at2m+18)/ '****'	/
		DATA qualifiers  (nqual_at2m+19)/ '****'	/
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
* SHOW ATTRIBUTE
	   DATA subcommands      (nsub_at2 +17)	/ 'ATTRIBUT'	/
	   DATA num_qualifs      (nsub_at2 +17)	/ nqual2r	/
	   DATA qualifier_pointer(nsub_at2 +17)	/ nqual_at2r	/
* SHOW ATTRIBUTE / ALL /DATASET /OUTPUT
		DATA qualifiers  (nqual_at2r+0)	/ 'ALL'		/
		DATA qualifiers  (nqual_at2r+1)	/ 'DATASET'	/
		DATA qualifiers  (nqual_at2r+2)	/ 'OUTPUT'	/
* SHOW NCCACHE
	   DATA subcommands      (nsub_at2 +18)	/ 'NCCACHE'	/
	   DATA num_qualifs      (nsub_at2 +18)	/ nqual2s	/
	   DATA qualifier_pointer(nsub_at2 +18)	/ nqual_at2s	/
* SHOW NCCACHE/ ****
		DATA qualifiers  (nqual_at2s+0)	/ '****'	/
* SHOW GIFFILE
	   DATA subcommands      (nsub_at2 +19)	/ 'GIFFILE'	/
	   DATA num_qualifs      (nsub_at2 +19)	/ nqual2t	/
	   DATA qualifier_pointer(nsub_at2 +19)	/ nqual_at2t	/
* SHOW GIFFILE/ ****
		DATA qualifiers  (nqual_at2t+0)	/ '****'	/
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

* SHOW TEXT         ! view text properties for PyFerret fonts
           DATA subcommands      (nsub_at2 +21)/ 'TEXT'	/
           DATA num_qualifs      (nsub_at2 +21)	/ nqual2v	/
           DATA qualifier_pointer(nsub_at2 +21)	/ nqual_at2v	/
* SHOW TEXT  / ****
                DATA qualifiers  (nqual_at2v+0)	/ '****'	/
                DATA qualifiers  (nqual_at2v+1)	/ '****'	/

* SHOW QUERIES
	   DATA subcommands      (nsub_at2 +22)	/ 'QUERIES'	/
	   DATA num_qualifs      (nsub_at2 +22)	/ nqual2w	/
	   DATA qualifier_pointer(nsub_at2 +22)	/ nqual_at2w	/
* SHOW QUERIES / ALL
		DATA qualifiers  (nqual_at2w+0)	/ 'ALL'		/
		DATA qualifiers  (nqual_at2w+1)	/ '****'	/
		DATA qualifiers  (nqual_at2w+2)	/ '****'	/

* CANCEL	 - command # 3
	DATA commands(3)			/ 'CANCEL'	/
        DATA its_action_command(3)              / .FALSE.       /
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
* CANCEL REGIONS /ALL/I/J/K/L/M/N/X/Y/Z/T/E/F
		DATA qualifiers  (nqual_at3c+0)	/ 'ALL '	/
		DATA qualifiers  (nqual_at3c+1)	/ 'I   '	/
		DATA qualifiers  (nqual_at3c+2)	/ 'J   '	/
		DATA qualifiers  (nqual_at3c+3)	/ 'K   '	/
		DATA qualifiers  (nqual_at3c+4)	/ 'L   '	/
		DATA qualifiers  (nqual_at3c+5)	/ 'M   '	/
		DATA qualifiers  (nqual_at3c+6)	/ 'N   '	/
		DATA qualifiers  (nqual_at3c+7)	/ 'X   '	/
		DATA qualifiers  (nqual_at3c+8)	/ 'Y   '	/
		DATA qualifiers  (nqual_at3c+9)	/ 'Z   '	/
		DATA qualifiers  (nqual_at3c+10)/ 'T   '	/
		DATA qualifiers  (nqual_at3c+11)/ 'E   '	/
		DATA qualifiers  (nqual_at3c+12)/ 'F   '	/
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
		DATA qualifiers  (nqual_at3f+1)	/ 'PRECISION'	/
		DATA qualifiers  (nqual_at3f+2)	/ 'FILE'	/
		DATA qualifiers  (nqual_at3f+3)	/ 'FORMAT'	/
		DATA qualifiers  (nqual_at3f+4)	/ 'HEADING'	/
		DATA qualifiers  (nqual_at3f+5)	/ 'APPEND'	/
		DATA qualifiers  (nqual_at3f+6)	/ 'OUTTYPE'	/
* CANCEL DATA_SET
	   DATA subcommands      (nsub_at3 + 6)	/ 'DATA'	/
	   DATA num_qualifs      (nsub_at3 + 6)	/ nqual3g	/
	   DATA qualifier_pointer(nsub_at3 + 6)	/ nqual_at3g	/
* CANCEL DATA /ALL
		DATA qualifiers  (nqual_at3g+0)	/ 'ALL '	/
		DATA qualifiers  (nqual_at3g+1)	/ 'NOERROR'	/
		DATA qualifiers  (nqual_at3g+2)	/ 'FMASK'	/
		DATA qualifiers  (nqual_at3g+3)	/ 'SMASK'	/
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
		DATA qualifiers  (nqual_at3k+0)	/ 'ALL '	/
		DATA qualifiers  (nqual_at3k+1)	/ 'DATASET'	/
		DATA qualifiers  (nqual_at3k+2)	/ 'LAYERZ'	/
		DATA qualifiers  (nqual_at3k+3)	/ 'CURVILINEAR'	/
		DATA qualifiers  (nqual_at3k+4)	/ 'NOERROR'	/
		DATA qualifiers  (nqual_at3k+5)	/ '****'	/
* CANCEL AXIS
	   DATA subcommands      (nsub_at3 +11)	/ 'AXIS'	/
	   DATA num_qualifs      (nsub_at3 +11)	/ nqual3l	/
	   DATA qualifier_pointer(nsub_at3 +11)	/ nqual_at3l	/
		DATA qualifiers  (nqual_at3l+0)	/ 'MODULO'	/
		DATA qualifiers  (nqual_at3l+1)	/ 'ALL'         /
		DATA qualifiers  (nqual_at3l+2)	/ 'DEPTH'       /
		DATA qualifiers  (nqual_at3l+3) / 'STRIDE'      /
		DATA qualifiers  (nqual_at3l+4) / '****'        /
		DATA qualifiers  (nqual_at3l+5) / '****'        /
* CANCEL GRID
	   DATA subcommands      (nsub_at3 +12)	/ 'GRID'	/
	   DATA num_qualifs      (nsub_at3 +12)	/ nqual3m	/
	   DATA qualifier_pointer(nsub_at3 +12)	/ nqual_at3m	/
		DATA qualifiers  (nqual_at3m+0)	/ '****'	/
* CANCEL ATTRIBUTE /OUTPUT /DATASET
	   DATA subcommands      (nsub_at3 +13)	/ 'ATTRIBUT'	/
	   DATA num_qualifs      (nsub_at3 +13)	/ nqual3n	/
	   DATA qualifier_pointer(nsub_at3 +13)	/ nqual_at3n	/
		DATA qualifiers  (nqual_at3n+0)	/ 'OUTPUT'	/
		DATA qualifiers  (nqual_at3n+1)	/ 'DATASET'	/
* CANCEL REDIRECT
	   DATA subcommands      (nsub_at3 +14)	/ 'REDIRECT'	/
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
* CANCEL NCCACHE
	   DATA subcommands      (nsub_at3 +17)	/ 'NCCACHE'	/
	   DATA num_qualifs      (nsub_at3 +17)	/ nqual3r	/
	   DATA qualifier_pointer(nsub_at3 +17)	/ nqual_at3r	/
		DATA qualifiers  (nqual_at3r+0)	/ '****'	/
* CANCEL PYVAR
	   DATA subcommands      (nsub_at3 +18)	/ 'PYVAR'	/
	   DATA num_qualifs      (nsub_at3 +18)	/ nqual3s	/
	   DATA qualifier_pointer(nsub_at3 +18)	/ nqual_at3s	/
		DATA qualifiers  (nqual_at3s+0)	/ 'ALL'		/
		DATA qualifiers  (nqual_at3s+1)	/ 'DATASET'	/
		DATA qualifiers  (nqual_at3s+2)	/ '****'	/
* CANCEL TEXT				                          ! unset text properties for PyFerret fonts
	   DATA subcommands      (nsub_at3 +19)	/ 'TEXT'	/
	   DATA num_qualifs      (nsub_at3 +19)	/ nqual3t	/
	   DATA qualifier_pointer(nsub_at3 +19)	/ nqual_at3t	/
		DATA qualifiers  (nqual_at3t+0)	/ 'FONT'	/
		DATA qualifiers  (nqual_at3t+1)	/ 'COLOR'	/
		DATA qualifiers  (nqual_at3t+2)	/ 'ITALIC'	/
		DATA qualifiers  (nqual_at3t+3)	/ 'BOLD'	/
		DATA qualifiers  (nqual_at3t+4)	/ 'ISIZ'	/
		DATA qualifiers  (nqual_at3t+5)	/ 'ALL'		/
		DATA qualifiers  (nqual_at3t+6)	/ '****'	/

* CONTOUR - command # 4
	DATA commands(4)			/ 'CONTOUR'	/
        DATA its_action_command(4)              / .TRUE.        /
	DATA num_subcommands(4)			/ nsub4		/
	DATA subcommand_pointer(4)		/ nsub_at4	/
* CONTOUR '    '
	   DATA subcommands      (nsub_at4 + 0)	/ '    '	/
	   DATA num_qualifs      (nsub_at4 + 0)	/ nqual4a	/
	   DATA qualifier_pointer(nsub_at4 + 0)	/ nqual_at4a	/
* CONTOUR/I/J/K/L/M/N/X/Y/Z/T/E/F/DATASET/OVERLAY/SET_UP/FRAME/TRANSPOSE
		DATA qualifiers  (nqual_at4a+0)	/ 'I'	/
		DATA qualifiers  (nqual_at4a+1)	/ 'J'	/
		DATA qualifiers  (nqual_at4a+2)	/ 'K'	/
		DATA qualifiers  (nqual_at4a+3)	/ 'L'	/
		DATA qualifiers  (nqual_at4a+4)	/ 'M'	/
		DATA qualifiers  (nqual_at4a+5)	/ 'N'	/
		DATA qualifiers  (nqual_at4a+6)	/ 'X'	/
		DATA qualifiers  (nqual_at4a+7)	/ 'Y'	/
		DATA qualifiers  (nqual_at4a+8)	/ 'Z'	/
		DATA qualifiers  (nqual_at4a+9)	/ 'T'	/
		DATA qualifiers  (nqual_at4a+10)/ 'E'	/
		DATA qualifiers  (nqual_at4a+11)/ 'F'	/
		DATA qualifiers  (nqual_at4a+12)/ 'OVERLAY'/	!/overlays
		DATA qualifiers  (nqual_at4a+13)/ 'SET_UP'/	!/set_up
		DATA qualifiers  (nqual_at4a+14)/ 'FRAME'/	!/frame
		DATA qualifiers  (nqual_at4a+15)/ 'DATASET'   /	!/data_set
		DATA qualifiers  (nqual_at4a+16)/ 'TRANPOSE'/	!/transpose
		DATA qualifiers  (nqual_at4a+17)/ 'FILL'/	!
		DATA qualifiers  (nqual_at4a+18)/ 'LINE'/	!
		DATA qualifiers  (nqual_at4a+19)/ 'NOLABEL'/	!/nolabel
		DATA qualifiers  (nqual_at4a+20)/ 'LEVELS'/	!/levels_same
		DATA qualifiers  (nqual_at4a+21)/ 'KEY' /	!
		DATA qualifiers  (nqual_at4a+22)/ 'NOKEY'/	!
		DATA qualifiers  (nqual_at4a+23)/ 'PALETTE'/	!
		DATA qualifiers  (nqual_at4a+24)/ 'XLIMITS'/	!/xlimits
		DATA qualifiers  (nqual_at4a+25)/ 'YLIMITS'/	!/ylimits
		DATA qualifiers  (nqual_at4a+26)/ 'TITLE'/	!/title
		DATA qualifiers  (nqual_at4a+27)/ 'COLOR'/	!/color
		DATA qualifiers  (nqual_at4a+28)/ 'NOAXES'/	!/noaxes
		DATA qualifiers  (nqual_at4a+29)/ 'PATTERN'/	!/pattern=
		DATA qualifiers  (nqual_at4a+30)/ 'SIZE'/	!/size
		DATA qualifiers  (nqual_at4a+31)/ 'SPACING'/	!/spacing
		DATA qualifiers  (nqual_at4a+32)/ 'SIGDIG'/	!/sigdig
		DATA qualifiers  (nqual_at4a+33)/ 'PEN'/	!/pen (denig!!)
		DATA qualifiers  (nqual_at4a+34)/ 'HLIMITS'/	! horiz
		DATA qualifiers  (nqual_at4a+35)/ 'VLIMITS'/	! vertical
		DATA qualifiers  (nqual_at4a+36)/ '****'/	! HLOG (place holder, not implemented for contuor) 
		DATA qualifiers  (nqual_at4a+37)/ '****'/	! VLOG (place holder)
		DATA qualifiers  (nqual_at4a+38)/ 'AXES'/	! /axes[=...]
		DATA qualifiers  (nqual_at4a+39)/ 'HGRATICU'/	! tic liness, dash or line
		DATA qualifiers  (nqual_at4a+40)/ 'VGRATICU'/	! tic liness, color or pen#
		DATA qualifiers  (nqual_at4a+41)/ 'GRATICUL'/	! 
		DATA qualifiers  (nqual_at4a+42)/ 'MODULO'/	! modulo for curvilinear plots
		DATA qualifiers  (nqual_at4a+43)/ 'THICKNES'/	! pen thickness
		DATA qualifiers  (nqual_at4a+44)/ 'OPACITY'/	! overriding oapcity
		DATA qualifiers  (nqual_at4a+45)/ 'DEGMINSEC'/	! Request deg/min or deg/min/sec axis labels
		DATA qualifiers  (nqual_at4a+46)/ 'HALFSPAC'/	! Request a space before the E/W or N/S on axis labels
		DATA qualifiers  (nqual_at4a+47)/ '****'/	! 

* LIST	 - command # 5
	DATA commands(5)			/ 'LIST'	/
        DATA its_action_command(5)              / .TRUE.        /
	DATA num_subcommands(5)			/ nsub5		/
	DATA subcommand_pointer(5)		/ nsub_at5	/
* LIST '    '
	   DATA subcommands      (nsub_at5 + 0)	/ '    '	/
	   DATA num_qualifs      (nsub_at5 + 0)	/ nqual5a	/
	   DATA qualifier_pointer(nsub_at5 + 0)	/ nqual_at5a	/
* LIST/I/J/K/L/M/N/X/Y/Z/T/E/F/D /HEAD /NOHEAD /G /FILE /APPEND /ORDER /FORMAT /QUIET 
		DATA qualifiers  (nqual_at5a+0)	/ 'I'	/
		DATA qualifiers  (nqual_at5a+1)	/ 'J'	/
		DATA qualifiers  (nqual_at5a+2)	/ 'K'	/
		DATA qualifiers  (nqual_at5a+3)	/ 'L'	/
		DATA qualifiers  (nqual_at5a+4)	/ 'M'	/
		DATA qualifiers  (nqual_at5a+5)	/ 'N'	/
		DATA qualifiers  (nqual_at5a+6)	/ 'X'	/
		DATA qualifiers  (nqual_at5a+7)	/ 'Y'	/
		DATA qualifiers  (nqual_at5a+8)	/ 'Z'	/
		DATA qualifiers  (nqual_at5a+9)	/ 'T'	/
		DATA qualifiers  (nqual_at5a+10)/ 'E'	/
		DATA qualifiers  (nqual_at5a+11)/ 'F'	/
		DATA qualifiers  (nqual_at5a+12)/ 'DATASET'	/
		DATA qualifiers  (nqual_at5a+13)/ 'HEADING'/
		DATA qualifiers  (nqual_at5a+14)/ 'NOHEAD'/
		DATA qualifiers  (nqual_at5a+15)/ 'SINGLY'/
		DATA qualifiers  (nqual_at5a+16)/ 'FILE'/
		DATA qualifiers  (nqual_at5a+17)/ 'APPEND'/
		DATA qualifiers  (nqual_at5a+18)/ 'ORDER'/
		DATA qualifiers  (nqual_at5a+19)/ 'FORMAT'/
		DATA qualifiers  (nqual_at5a+20)/ 'TITLE'/
		DATA qualifiers  (nqual_at5a+21)/ 'PRECISION'/ !used to be EPKEY 
		DATA qualifiers  (nqual_at5a+22)/ 'RIGID'/ ! /RIGID (CDF)
		DATA qualifiers  (nqual_at5a+23)/ 'ILIMITS'/ 
		DATA qualifiers  (nqual_at5a+24)/ 'JLIMITS'/
		DATA qualifiers  (nqual_at5a+25)/ 'KLIMITS'/
		DATA qualifiers  (nqual_at5a+26)/ 'LLIMITS'/
		DATA qualifiers  (nqual_at5a+27)/ 'MLIMITS'/
		DATA qualifiers  (nqual_at5a+28)/ 'NLIMITS'/
		DATA qualifiers  (nqual_at5a+29)/ 'XLIMITS'/
		DATA qualifiers  (nqual_at5a+30)/ 'YLIMITS'/
		DATA qualifiers  (nqual_at5a+31)/ 'ZLIMITS'/
		DATA qualifiers  (nqual_at5a+32)/ 'TLIMITS'/
		DATA qualifiers  (nqual_at5a+33)/ 'ELIMITS'/
		DATA qualifiers  (nqual_at5a+34)/ 'FLIMITS'/
		DATA qualifiers  (nqual_at5a+35)/ 'CLOBBER'/
		DATA qualifiers  (nqual_at5a+36)/ 'QUIET'/
		DATA qualifiers  (nqual_at5a+37)/ 'WIDTH'/
		DATA qualifiers  (nqual_at5a+38)/ 'EDGES'/
		DATA qualifiers  (nqual_at5a+39)/ 'BOUNDS'/
		DATA qualifiers  (nqual_at5a+40)/ 'NOBOUNDS'/
		DATA qualifiers  (nqual_at5a+41)/ 'NOROWLAB'/
		DATA qualifiers  (nqual_at5a+42)/ 'KEEP_AXI'/
		DATA qualifiers  (nqual_at5a+43)/ 'NCFORMAT'/
		DATA qualifiers  (nqual_at5a+44)/ 'XCHUNK'/
		DATA qualifiers  (nqual_at5a+45)/ 'YCHUNK'/
		DATA qualifiers  (nqual_at5a+46)/ 'ZCHUNK'/
		DATA qualifiers  (nqual_at5a+47)/ 'TCHUNK'/
		DATA qualifiers  (nqual_at5a+48)/ 'ECHUNK'/
		DATA qualifiers  (nqual_at5a+49)/ 'FCHUNK'/
		DATA qualifiers  (nqual_at5a+50)/ 'DEFLATE'/
		DATA qualifiers  (nqual_at5a+51)/ 'SHUFFLE'/
		DATA qualifiers  (nqual_at5a+52)/ 'ENDIAN'/
		DATA qualifiers  (nqual_at5a+53)/ 'OUTTYPE'/
		DATA qualifiers  (nqual_at5a+54)/ 'CURVILIN'/
		DATA qualifiers  (nqual_at5a+55)/ 'LAYERZ'/
		DATA qualifiers  (nqual_at5a+56)/ 'ASGLOBAL'/
		DATA qualifiers  (nqual_at5a+57)/ 'STRLEN'/   ! user-set length of string-dimension
		DATA qualifiers  (nqual_at5a+58)/ 'NOCOORDS'/ ! write dimensions only not coord vars. 
		DATA qualifiers  (nqual_at5a+59)/ '****'/

* PLOT - command # 6
	DATA commands(6)			/ 'PLOT'	/
        DATA its_action_command(6)              / .TRUE.        /
	DATA num_subcommands(6)			/ nsub6		/
	DATA subcommand_pointer(6)		/ nsub_at6	/
* PLOT '    '
	   DATA subcommands      (nsub_at6 + 0)	/ '    '	/
	   DATA num_qualifs      (nsub_at6 + 0)	/ nqual6a	/
	   DATA qualifier_pointer(nsub_at6 + 0)	/ nqual_at6a	/
* PLOT/I/J/K/L/M/N/X/Y/Z/T/E/F/DATASET/OVERLAY/SET_UP/FRAME/TRANSPOSE/VS
		DATA qualifiers  (nqual_at6a+0)	/ 'I'	/
		DATA qualifiers  (nqual_at6a+1)	/ 'J'	/
		DATA qualifiers  (nqual_at6a+2)	/ 'K'	/
		DATA qualifiers  (nqual_at6a+3)	/ 'L'	/
		DATA qualifiers  (nqual_at6a+4)	/ 'M'	/
		DATA qualifiers  (nqual_at6a+5)	/ 'N'	/
		DATA qualifiers  (nqual_at6a+6)	/ 'X'	/
		DATA qualifiers  (nqual_at6a+7)	/ 'Y'	/
		DATA qualifiers  (nqual_at6a+8)	/ 'Z'	/
		DATA qualifiers  (nqual_at6a+9) / 'T'	/
		DATA qualifiers  (nqual_at6a+10)/ 'E'	/
		DATA qualifiers  (nqual_at6a+11)/ 'F'	/
		DATA qualifiers  (nqual_at6a+12)/ 'ALONG'/	!/along
		DATA qualifiers  (nqual_at6a+13)/ 'OVERLAY'/	!/overlays
		DATA qualifiers  (nqual_at6a+14)/ 'SET_UP'/	!/set_up
		DATA qualifiers  (nqual_at6a+15)/ 'FRAME'/	!/frame
		DATA qualifiers  (nqual_at6a+16)/ 'DATASET'   /	!/data_set
		DATA qualifiers  (nqual_at6a+17)/ 'TRANPOSE'/	!/transpose
		DATA qualifiers  (nqual_at6a+18)/ 'VS  '/	!/versus
		DATA qualifiers  (nqual_at6a+19)/ 'SYMBOL'/	!/symbol
		DATA qualifiers  (nqual_at6a+20)/ 'NOLABEL'/	!/nolabel
		DATA qualifiers  (nqual_at6a+21)/ 'LINE'/	!/line
		DATA qualifiers  (nqual_at6a+22)/ 'COLOR'/	!/color
		DATA qualifiers  (nqual_at6a+23)/ 'THICKNES'/	!/thickness
		DATA qualifiers  (nqual_at6a+24)/ 'XLIMITS'/	!/xlimits
		DATA qualifiers  (nqual_at6a+25)/ 'YLIMITS'/	!/ylimits
		DATA qualifiers  (nqual_at6a+26)/ 'TITLE'/	!/title
		DATA qualifiers  (nqual_at6a+27)/ 'SIZE'/	!/size
		DATA qualifiers  (nqual_at6a+28)/ 'NOAXES'/	!/noaxes
		DATA qualifiers  (nqual_at6a+29)/ 'STEP'/	!/step
		DATA qualifiers  (nqual_at6a+30)/ 'DASH'/	!/dash
		DATA qualifiers  (nqual_at6a+31)/ 'AXES'/	!/axes[=]
		DATA qualifiers  (nqual_at6a+32)/ 'HGRATICU'/	!horiz tic lines
		DATA qualifiers  (nqual_at6a+33)/ 'VGRATICU'/	!vert tic lines
		DATA qualifiers  (nqual_at6a+34)/ 'HLIMITS'/	! horiz
		DATA qualifiers  (nqual_at6a+35)/ 'VLIMITS'/	! vertical
		DATA qualifiers  (nqual_at6a+36)/ 'HLOG'/	! log axis horiz
		DATA qualifiers  (nqual_at6a+37)/ 'VLOG'/	! log axis vert
		DATA qualifiers  (nqual_at6a+38)/ 'GRATICUL'/	! tic lines
		DATA qualifiers  (nqual_at6a+39)/ 'NOKEY'/	! nokey
		DATA qualifiers  (nqual_at6a+40)/ 'NOYADJUS'/	! noyadjust
		DATA qualifiers  (nqual_at6a+41)/ 'KEY'/	! [=cont] for Ribbon plot
		DATA qualifiers  (nqual_at6a+42)/ 'RIBBON'/	! for Ribbon plot
		DATA qualifiers  (nqual_at6a+43)/ 'LEVELS'/	! 
		DATA qualifiers  (nqual_at6a+44)/ 'PALETTE'/	! 
		DATA qualifiers  (nqual_at6a+45)/ 'FAST'/	! style for Ribbon plot
		DATA qualifiers  (nqual_at6a+46)/ 'MISSING'/	! For missing-dat in ribbon plot
		DATA qualifiers  (nqual_at6a+47)/ 'OPACITY'/	! overriding opacity
		DATA qualifiers  (nqual_at6a+48)/ 'GAPLOC'/     ! A variable marking gaps to insert in PLOT/VS
		DATA qualifiers  (nqual_at6a+49)/ 'SKIPSYM'/	! Control for symbol decimation
		DATA qualifiers  (nqual_at6a+50)/ 'DEGMINSEC'/	! Request deg/min or deg/min/sec axis labels
		DATA qualifiers  (nqual_at6a+51)/ 'HALFSPAC'/	! Request a space before the E/W or N/S on axis labels
		DATA qualifiers  (nqual_at6a+52)/ 'HIGHLIGH'/	! HIGHLIGHT in PyFerret
		DATA qualifiers  (nqual_at6a+53)/ '****'/	! unused
		DATA qualifiers  (nqual_at6a+54)/ '****'/	! unused
		DATA qualifiers  (nqual_at6a+55)/ '****'/	! unused

* GO	 - command # 7
	DATA commands(7)			/ 'GO  '	/
        DATA its_action_command(7)              / .FALSE.       /
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
        DATA its_action_command(8)              / .FALSE.       /
	DATA num_subcommands(8)			/ nsub8		/
	DATA subcommand_pointer(8)		/ nsub_at8	/
* HELP '    '
	   DATA subcommands      (nsub_at8 + 0)	/ '    '	/
	   DATA num_qualifs      (nsub_at8 + 0)	/ nqual8a	/
	   DATA qualifier_pointer(nsub_at8 + 0)	/ nqual_at8a	/

* LOAD - command # 9
	DATA commands(9)			/ 'LOAD'	/
        DATA its_action_command(9)              / .TRUE.        /
	DATA num_subcommands(9)			/ nsub9		/
	DATA subcommand_pointer(9)		/ nsub_at9	/
* LOAD '    '
	   DATA subcommands      (nsub_at9 + 0)	/ '    '	/
	   DATA num_qualifs      (nsub_at9 + 0)	/ nqual9a	/
	   DATA qualifier_pointer(nsub_at9 + 0)	/ nqual_at9a	/
* LOAD /TEMPORARY /PERMANENT /I/J/K/L/M/N/X/Y/Z/T/E/F/D /NAME
		DATA qualifiers  (nqual_at9a+0)	/ 'TEMPORY'/
		DATA qualifiers  (nqual_at9a+1)	/ 'PERMANT'/
		DATA qualifiers  (nqual_at9a+2)	/ 'I'	/
		DATA qualifiers  (nqual_at9a+3)	/ 'J'	/
		DATA qualifiers  (nqual_at9a+4)	/ 'K'	/
		DATA qualifiers  (nqual_at9a+5)	/ 'L'	/
		DATA qualifiers  (nqual_at9a+6)	/ 'M'	/
		DATA qualifiers  (nqual_at9a+7)	/ 'N'	/
		DATA qualifiers  (nqual_at9a+8)	/ 'X'	/
		DATA qualifiers  (nqual_at9a+9)	/ 'Y'	/
		DATA qualifiers  (nqual_at9a+10)/ 'Z'	/
		DATA qualifiers  (nqual_at9a+11)/ 'T'	/
		DATA qualifiers  (nqual_at9a+12)/ 'E'	/
		DATA qualifiers  (nqual_at9a+13)/ 'F'	/
		DATA qualifiers  (nqual_at9a+14)/ 'DATASET'	/
		DATA qualifiers  (nqual_at9a+15)/ 'NAME'/

* DEFINE - command # 10
	DATA commands(10)			/ 'DEFINE'	/
        DATA its_action_command(10)             / .FALSE.       /
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
* DEFINE REGION /I/J/K/L/M/N/X/Y/Z/T/E/F /DEFAULT / ****
		DATA qualifiers  (nqual_at10b+0)/ 'I'	/
		DATA qualifiers  (nqual_at10b+1)/ 'J'	/
		DATA qualifiers  (nqual_at10b+2)/ 'K'	/
		DATA qualifiers  (nqual_at10b+3)/ 'L'	/
		DATA qualifiers  (nqual_at10b+4)/ 'M'	/
		DATA qualifiers  (nqual_at10b+5)/ 'N'	/
		DATA qualifiers  (nqual_at10b+6)/ 'X'	/
		DATA qualifiers  (nqual_at10b+7)/ 'Y'	/
		DATA qualifiers  (nqual_at10b+8)/ 'Z'	/
		DATA qualifiers  (nqual_at10b+9)/ 'T'	/
		DATA qualifiers  (nqual_at10b+10)/ 'E'	/
		DATA qualifiers  (nqual_at10b+11)/ 'F'	/
		DATA qualifiers  (nqual_at10b+12)/ 'DEFAULT'/
		DATA qualifiers  (nqual_at10b+13)/ '****'/
		DATA qualifiers  (nqual_at10b+14)/ 'DX'/
		DATA qualifiers  (nqual_at10b+15)/ 'DY'/
		DATA qualifiers  (nqual_at10b+16)/ 'DZ'/
		DATA qualifiers  (nqual_at10b+17)/ 'DT'/
		DATA qualifiers  (nqual_at10b+18)/ 'DE'/
		DATA qualifiers  (nqual_at10b+19)/ 'DF'/
		DATA qualifiers  (nqual_at10b+20)/ 'DI'/
		DATA qualifiers  (nqual_at10b+21)/ 'DJ'/
		DATA qualifiers  (nqual_at10b+22)/ 'DK'/
		DATA qualifiers  (nqual_at10b+23)/ 'DL'/
		DATA qualifiers  (nqual_at10b+24)/ 'DM'/
		DATA qualifiers  (nqual_at10b+25)/ 'DN'/
* DEFINE GRID
	   DATA subcommands      (nsub_at10 + 2)/ 'GRID'	/
	   DATA num_qualifs      (nsub_at10 + 2)/ nqual10c	/
	   DATA qualifier_pointer(nsub_at10 + 2)/ nqual_at10c	/
* DEFINE GRID /X/Y/Z/T/E/F /FILE /LIKE
		DATA qualifiers  (nqual_at10c+ 0)/ 'X'	  /
		DATA qualifiers  (nqual_at10c+ 1)/ 'Y'	  /
		DATA qualifiers  (nqual_at10c+ 2)/ 'Z'	  /
		DATA qualifiers  (nqual_at10c+ 3)/ 'T'	  /
		DATA qualifiers  (nqual_at10c+ 4)/ 'E'	  /
		DATA qualifiers  (nqual_at10c+ 5)/ 'F'	  /
		DATA qualifiers  (nqual_at10c+ 6)/ 'FILE' /
		DATA qualifiers  (nqual_at10c+ 7)/ 'LIKE' /
		DATA qualifiers  (nqual_at10c+ 8)/ '****' /
		DATA qualifiers  (nqual_at10c+ 9)/ '****' /
		DATA qualifiers  (nqual_at10c+10)/ '****' /
* DEFINE VARIABLE
	   DATA subcommands      (nsub_at10 + 3)/ 'VARIABLE'	/
	   DATA num_qualifs      (nsub_at10 + 3)/ nqual10d	/
	   DATA qualifier_pointer(nsub_at10 + 3)/ nqual_at10d	/
* DEFINE VARIABLE /TITLE /UNITS /QUIET /DATASET /BAD /REMOTE /LIKE
		DATA qualifiers  (nqual_at10d+ 0)/ 'TITLE' /
		DATA qualifiers  (nqual_at10d+ 1)/ 'UNITS' /
		DATA qualifiers  (nqual_at10d+ 2)/ 'QUIET' /
		DATA qualifiers  (nqual_at10d+ 3)/ 'DATASET' /
		DATA qualifiers  (nqual_at10d+ 4)/ 'BAD' /
		DATA qualifiers  (nqual_at10d+ 5)/ 'REMOTE' /
		DATA qualifiers  (nqual_at10d+ 6)/ 'LIKE' /
		DATA qualifiers  (nqual_at10d+ 7)/ '****' /
		DATA qualifiers  (nqual_at10d+ 8)/ '****' /
		DATA qualifiers  (nqual_at10d+ 9)/ '****' /
* DEFINE AXIS
	   DATA subcommands      (nsub_at10 + 4)/ 'AXIS'	/
	   DATA num_qualifs      (nsub_at10 + 4)/ nqual10e	/
	   DATA qualifier_pointer(nsub_at10 + 4)/ nqual_at10e	/
* DEFINE AXIS /X/Y/Z/T/E/F /FILE /UNITS /QUIET /MONTHLY /LIKE ...
		DATA qualifiers  (nqual_at10e+ 0)/ 'X'	  /
		DATA qualifiers  (nqual_at10e+ 1)/ 'Y'	  /
		DATA qualifiers  (nqual_at10e+ 2)/ 'Z'	  /
		DATA qualifiers  (nqual_at10e+ 3)/ 'T'	  /
		DATA qualifiers  (nqual_at10e+ 4)/ 'E'	  /
		DATA qualifiers  (nqual_at10e+ 5)/ 'F'	  /
		DATA qualifiers  (nqual_at10e+ 6)/ 'FILE' /
		DATA qualifiers  (nqual_at10e+ 7)/ 'UNITS' /
		DATA qualifiers  (nqual_at10e+ 8)/ 'T0' /
		DATA qualifiers  (nqual_at10e+ 9)/ 'NAME' /
		DATA qualifiers  (nqual_at10e+ 10)/ 'FROMDATA' /
		DATA qualifiers  (nqual_at10e+ 11)/ 'DEPTH' /
		DATA qualifiers  (nqual_at10e+ 12)/ 'MODULO' /
		DATA qualifiers  (nqual_at10e+ 13)/ 'NPOINTS' /
		DATA qualifiers  (nqual_at10e+ 14)/ 'EDGES' /
		DATA qualifiers  (nqual_at10e+ 15)/ 'CALENDAR' /
		DATA qualifiers  (nqual_at10e+ 16)/ 'BOUNDS' /
		DATA qualifiers  (nqual_at10e+ 17)/ 'QUIET' /
		DATA qualifiers  (nqual_at10e+ 18)/ 'MONTHLY' /
		DATA qualifiers  (nqual_at10e+ 19)/ 'LIKE' /
* DEFINE VIEW
	   DATA subcommands      (nsub_at10 + 5)/ 'VIEWPORT'	/
	   DATA num_qualifs      (nsub_at10 + 5)/ nqual10f	/
	   DATA qualifier_pointer(nsub_at10 + 5)/ nqual_at10f	/
* DEFINE VIEW /SIZE/TEXT/XLIMITS/YLIMITS/AXES ****
		DATA qualifiers  (nqual_at10f+ 0)/ 'TEXT' /
		DATA qualifiers  (nqual_at10f+ 1)/ 'XLIMITS' /
		DATA qualifiers  (nqual_at10f+ 2)/ 'YLIMITS' /
		DATA qualifiers  (nqual_at10f+ 3)/ 'SIZE' /
		DATA qualifiers  (nqual_at10f+ 4)/ 'ORIGIN' /
		DATA qualifiers  (nqual_at10f+ 5)/ 'CLIP' /
		DATA qualifiers  (nqual_at10f+ 6)/ 'AXES' /
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
* DEFINE ATTRIBUTE
	   DATA subcommands      (nsub_at10 + 8)/ 'ATTRIBUT'	/
	   DATA num_qualifs      (nsub_at10 + 8)/ nqual10i	/
	   DATA qualifier_pointer(nsub_at10 + 8)/ nqual_at10i	/
* DEFINE ATTRIBUTE /DATASET /TYPE /OUTPUT /QUIET 
		DATA qualifiers  (nqual_at10i+ 0)/ 'DATASET' /
		DATA qualifiers  (nqual_at10i+ 1)/ 'TYPE' /
		DATA qualifiers  (nqual_at10i+ 2)/ 'OUTPUT' /
		DATA qualifiers  (nqual_at10i+ 3)/ 'QUIET' /
* DEFINE PYFUNC
	   DATA subcommands      (nsub_at10 + 9)/ 'PYFUNC'	/
	   DATA num_qualifs      (nsub_at10 + 9)/ nqual10j	/
	   DATA qualifier_pointer(nsub_at10 + 9)/ nqual_at10j	/
* DEFINE PYFUNC /NAME ****
		DATA qualifiers  (nqual_at10j+ 0)/ 'NAME' /
		DATA qualifiers  (nqual_at10j+ 1)/ '****' /
* DEFINE DATA
	   DATA subcommands      (nsub_at10 + 10)/ 'DATA'	/
	   DATA num_qualifs      (nsub_at10 + 10)/ nqual10k	/
	   DATA qualifier_pointer(nsub_at10 + 10)/ nqual_at10k	/
* DEFINE DATA / AGGREGATE /TITLE /QUIET /HIDE /T /E /F /U /TAXIS /FAXIS /TOVERLAP
		DATA qualifiers  (nqual_at10k+ 0)/ 'AGGREGATE' /
		DATA qualifiers  (nqual_at10k+ 1)/ '****' / ! reserved for x
		DATA qualifiers  (nqual_at10k+ 2)/ '****' / ! y
		DATA qualifiers  (nqual_at10k+ 3)/ '****' / ! z
		DATA qualifiers  (nqual_at10k+ 4)/ 'T'    / ! t
		DATA qualifiers  (nqual_at10k+ 5)/ 'E'    / ! e
		DATA qualifiers  (nqual_at10k+ 6)/ 'F'    / ! f
		DATA qualifiers  (nqual_at10k+ 7)/ 'TITLE' /
		DATA qualifiers  (nqual_at10k+ 8)/ 'QUIET' /
		DATA qualifiers  (nqual_at10k+ 9)/ 'HIDE' /
		DATA qualifiers  (nqual_at10k+ 10)/ 'U' /
		DATA qualifiers  (nqual_at10k+ 11)/ 'TAXIS' /
		DATA qualifiers  (nqual_at10k+ 12)/ 'FAXIS' /
		DATA qualifiers  (nqual_at10k+ 13)/ 'TOVERLAP' /
* DEFINE ANNOTATION
	   DATA subcommands      (nsub_at10 + 11)/ 'ANNOTATION'	/
	   DATA num_qualifs      (nsub_at10 + 11)/ nqual10l	/
	   DATA qualifier_pointer(nsub_at10 + 11)/ nqual_at10l	/
* DEFINE ANNOTATION /USER/NORM/NOUSER/XPOS/YPOS/HALIGN/VALIGN/ANGLE/SIZE/NLAB
		DATA qualifiers  (nqual_at10l+ 0)/ 'USER' /
		DATA qualifiers  (nqual_at10l+ 1)/ 'NORM' /
		DATA qualifiers  (nqual_at10l+ 2)/ 'NOUSER' /
		DATA qualifiers  (nqual_at10l+ 3)/ 'XPOS' /
		DATA qualifiers  (nqual_at10l+ 4)/ 'YPOS' /
		DATA qualifiers  (nqual_at10l+ 5)/ 'HALIGN' /
		DATA qualifiers  (nqual_at10l+ 6)/ 'VALIGN' /
		DATA qualifiers  (nqual_at10l+ 7)/ 'ANGLE' /
		DATA qualifiers  (nqual_at10l+ 8)/ 'SIZE' /
		DATA qualifiers  (nqual_at10l+ 9)/ 'NLAB' /
* EXIT	 - command # 11
	DATA commands(11)			/ 'EXIT'	/
        DATA its_action_command(11)             / .FALSE.       /
	DATA num_subcommands(11)		/ nsub11	/
	DATA subcommand_pointer(11)		/ nsub_at11	/
* EXIT '    '
	   DATA subcommands      (nsub_at11 + 0)/ '    '	/
	   DATA num_qualifs      (nsub_at11 + 0)/ nqual11a	/
	   DATA qualifier_pointer(nsub_at11 + 0)/ nqual_at11a	/
* EXIT/COMMAND_FILE etc
		DATA qualifiers  (nqual_at11a+0)/ 'COMMAND'	/
		DATA qualifiers  (nqual_at11a+1)/ 'LOOP'	/
		DATA qualifiers  (nqual_at11a+2)/ 'SCRIPT'	/
		DATA qualifiers  (nqual_at11a+3)/ 'PROMPT'	/
		DATA qualifiers  (nqual_at11a+4)/ 'PROGRAM'	/
		DATA qualifiers  (nqual_at11a+5)/ 'CYCLE'	/
		DATA qualifiers  (nqual_at11a+6)/ 'TOPYTHON'	/
		DATA qualifiers  (nqual_at11a+7)/ '****'	/

* MESSAGE	 - command # 12
	DATA commands(12)			/ 'MESSAGE'	/
        DATA its_action_command(12)             / .FALSE.       /
	DATA num_subcommands(12)		/ nsub12	/
	DATA subcommand_pointer(12)		/ nsub_at12	/
* MESSAGE '    '
	   DATA subcommands      (nsub_at12 + 0)/ '    '	/
	   DATA num_qualifs      (nsub_at12 + 0)/ nqual12a	/
	   DATA qualifier_pointer(nsub_at12 + 0)/ nqual_at12a	/
* MESSAGE/CONTINUE
		DATA qualifiers  (nqual_at12a+0)/ 'CONTINUE'	/
* MESSAGE/QUIET/JOURNAL/ERROR/OUTFILE/APPEND/CLOBBER
		DATA qualifiers  (nqual_at12a+1)/ 'QUIET'	/
		DATA qualifiers  (nqual_at12a+2)/ 'JOURNAL'	/
		DATA qualifiers  (nqual_at12a+3)/ 'ERROR'	/
		DATA qualifiers  (nqual_at12a+4)/ 'OUTFILE'	/
		DATA qualifiers  (nqual_at12a+5)/ 'APPEND'	/
		DATA qualifiers  (nqual_at12a+6)/ 'CLOBBER'	/

* VECTOR - command # 13
	DATA commands(13)			/ 'VECTOR'	/
        DATA its_action_command(13)             / .TRUE.        /
	DATA num_subcommands(13)		/ nsub13	/
	DATA subcommand_pointer(13)		/ nsub_at13	/
* VECTOR '    '
	   DATA subcommands      (nsub_at13 + 0)/ '    '	/
	   DATA num_qualifs      (nsub_at13 + 0)/ nqual13a	/
	   DATA qualifier_pointer(nsub_at13 + 0)/ nqual_at13a	/
* VECTOR/I/J/K/L/M/N/X/Y/Z/T/E/F/DATASET/OVERLAY/SET_UP/FRAME/TRANSPOSE
		DATA qualifiers  (nqual_at13a+0)/ 'I'	/
		DATA qualifiers  (nqual_at13a+1)/ 'J'	/
		DATA qualifiers  (nqual_at13a+2)/ 'K'	/
		DATA qualifiers  (nqual_at13a+3)/ 'L'	/
		DATA qualifiers  (nqual_at13a+4)/ 'M'	/
		DATA qualifiers  (nqual_at13a+5)/ 'N'	/
		DATA qualifiers  (nqual_at13a+6)/ 'X'	/
		DATA qualifiers  (nqual_at13a+7)/ 'Y'	/
		DATA qualifiers  (nqual_at13a+8)/ 'Z'	/
		DATA qualifiers  (nqual_at13a+9)/ 'T'	/
		DATA qualifiers  (nqual_at13a+10)/ 'E'	/
		DATA qualifiers  (nqual_at13a+11)/ 'F'	/
		DATA qualifiers  (nqual_at13a+12)/ 'OVERLAY'/	!/overlays
		DATA qualifiers  (nqual_at13a+13)/ 'SET_UP'/	!/set_up
		DATA qualifiers  (nqual_at13a+14)/ 'FRAME'/	!/frame
		DATA qualifiers  (nqual_at13a+15)/ 'DATASET' /	!/data_set
		DATA qualifiers  (nqual_at13a+16)/ 'TRANPOSE'/	!/transpose
		DATA qualifiers  (nqual_at13a+17)/ '****'/	!
		DATA qualifiers  (nqual_at13a+18)/ 'ASPECT'/	!/aspect
		DATA qualifiers  (nqual_at13a+19)/ 'NOLABEL'/	!/nolabel
		DATA qualifiers  (nqual_at13a+20)/ 'LENGTH'/	!/length_same
		DATA qualifiers  (nqual_at13a+21)/ 'XSKIP'/	!/xskip
		DATA qualifiers  (nqual_at13a+22)/ 'YSKIP'/	!/yskip
		DATA qualifiers  (nqual_at13a+23)/ 'THICK'/	!/thick
		DATA qualifiers  (nqual_at13a+24)/ 'XLIMITS'/	!/xlimits
		DATA qualifiers  (nqual_at13a+25)/ 'YLIMITS'/	!/ylimits
		DATA qualifiers  (nqual_at13a+26)/ 'TITLE'/	!/title
		DATA qualifiers  (nqual_at13a+27)/ 'COLOR'/	!/color
		DATA qualifiers  (nqual_at13a+28)/ 'NOAXES'/	!/noaxes
		DATA qualifiers  (nqual_at13a+29)/ 'NOKEY'/	!/nokey
		DATA qualifiers  (nqual_at13a+30)/ 'FLOWLINE'/	!/flowline
		DATA qualifiers  (nqual_at13a+31)/ 'DENSITY'/	!/density=
		DATA qualifiers  (nqual_at13a+32)/ 'AXES'/	!/axes[=]
		DATA qualifiers  (nqual_at13a+33)/ 'PEN'/	! pen (denigr!)
		DATA qualifiers  (nqual_at13a+34)/ 'HLIMITS'/	! horiz
		DATA qualifiers  (nqual_at13a+35)/ 'VLIMITS'/	! vertical
		DATA qualifiers  (nqual_at13a+36)/ 'HGRATICU'/	! horiz tic lines
		DATA qualifiers  (nqual_at13a+37)/ 'VGRATICU'/	! vert tic lines
		DATA qualifiers  (nqual_at13a+38)/ 'GRATICUL'/
		DATA qualifiers  (nqual_at13a+39)/ 'KEY'/	!KEY 
		DATA qualifiers  (nqual_at13a+40)/ 'MODULO'/
		DATA qualifiers  (nqual_at13a+41)/ 'OPACITY'/	! overriding opacity
		DATA qualifiers  (nqual_at13a+42)/ '****'/
		DATA qualifiers  (nqual_at13a+43)/ '****'/
		DATA qualifiers  (nqual_at13a+44)/ '****'/

* PPLUS	 - command # 14
	DATA commands(14)			/ 'PPLUS'	/
        DATA its_action_command(14)             / .FALSE.       /
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
        DATA its_action_command(15)             / .FALSE.       /
	DATA num_subcommands(15)		/ nsub15	/
	DATA subcommand_pointer(15)		/ nsub_at15	/
* FRAME '    '
	   DATA subcommands      (nsub_at15 + 0)/ '    '	/
	   DATA num_qualifs      (nsub_at15 + 0)/ nqual15a	/
	   DATA qualifier_pointer(nsub_at15 + 0)/ nqual_at15a	/
* FRAME/FORMAT=.../FILE=.../TRANSPARENT
*      /XINCHES=.../YINCHES=.../XPIXELS=.../YPIXELS=.../ANNOTATE=...
		DATA qualifiers  (nqual_at15a+0)/ 'FORMAT'	/
		DATA qualifiers  (nqual_at15a+1)/ 'FILE'	/
		DATA qualifiers  (nqual_at15a+2)/ 'TRANSPAR'	/
		DATA qualifiers  (nqual_at15a+3)/ 'XINCHES'	/
		DATA qualifiers  (nqual_at15a+4)/ 'YINCHES'	/
		DATA qualifiers  (nqual_at15a+5)/ 'XPIXELS'	/
		DATA qualifiers  (nqual_at15a+6)/ 'YPIXELS'	/
		DATA qualifiers  (nqual_at15a+7)/ 'ANNOTATE'	/

* REPEAT - command # 16
	DATA commands(16)			/ 'REPEAT'	/
        DATA its_action_command(16)             / .FALSE.       /
	DATA num_subcommands(16)		/ nsub16	/
	DATA subcommand_pointer(16)		/ nsub_at16	/
* REPEAT '    '
	   DATA subcommands      (nsub_at16 + 0)/ '    '	/
	   DATA num_qualifs      (nsub_at16 + 0)/ nqual16a	/
	   DATA qualifier_pointer(nsub_at16 + 0)/ nqual_at16a	/
* REPEAT/I/J/K/L/M/N/X/Y/Z/T/E/F/ANIMATE/LOOP=[N]
		DATA qualifiers  (nqual_at16a+0)/ 'I'	/
		DATA qualifiers  (nqual_at16a+1)/ 'J'	/
		DATA qualifiers  (nqual_at16a+2)/ 'K'	/
		DATA qualifiers  (nqual_at16a+3)/ 'L'	/
		DATA qualifiers  (nqual_at16a+4)/ 'M'	/
		DATA qualifiers  (nqual_at16a+5)/ 'N'	/
		DATA qualifiers  (nqual_at16a+6)/ 'X'	/
		DATA qualifiers  (nqual_at16a+7)/ 'Y'	/
		DATA qualifiers  (nqual_at16a+8)/ 'Z'	/
		DATA qualifiers  (nqual_at16a+9)/ 'T'	/
		DATA qualifiers  (nqual_at16a+10)/ 'E'	/
		DATA qualifiers  (nqual_at16a+11)/ 'F'	/
		DATA qualifiers  (nqual_at16a+12)/ 'ANIMATE' /
		DATA qualifiers  (nqual_at16a+13)/ 'LOOP'  /
		DATA qualifiers  (nqual_at16a+14)/ 'RANGE' /   
		DATA qualifiers  (nqual_at16a+15)/ 'NAME' /   
		DATA qualifiers  (nqual_at16a+16)/ '****' /  

* STATISTICS - command # 17
	DATA commands(17)			/ 'STAT'	/
        DATA its_action_command(17)             / .TRUE.        /
	DATA num_subcommands(17)		/ nsub17	/
	DATA subcommand_pointer(17)		/ nsub_at17	/
* STAT '    '
	   DATA subcommands      (nsub_at17 + 0)	/ '    '	/
	   DATA num_qualifs      (nsub_at17 + 0)	/ nqual17a	/
	   DATA qualifier_pointer(nsub_at17 + 0)	/ nqual_at17a	/
* STAT /BRIEF/PRECISION/I/J/K/L/M/N/X/Y/Z/T/E/F/DATASET/G
		DATA qualifiers  (nqual_at17a+0)	/ 'BRIEF'/
		DATA qualifiers  (nqual_at17a+1)	/ 'PRECISION'/
		DATA qualifiers  (nqual_at17a+2)	/ 'I'	/
		DATA qualifiers  (nqual_at17a+3)	/ 'J'	/
		DATA qualifiers  (nqual_at17a+4)	/ 'K'	/
		DATA qualifiers  (nqual_at17a+5)	/ 'L'	/
		DATA qualifiers  (nqual_at17a+6)	/ 'M'	/
		DATA qualifiers  (nqual_at17a+7)	/ 'N'	/
		DATA qualifiers  (nqual_at17a+8)	/ 'X'	/
		DATA qualifiers  (nqual_at17a+9)	/ 'Y'	/
		DATA qualifiers  (nqual_at17a+10)	/ 'Z'	/
		DATA qualifiers  (nqual_at17a+11)	/ 'T'	/
		DATA qualifiers  (nqual_at17a+12)	/ 'E'	/
		DATA qualifiers  (nqual_at17a+13)	/ 'F'	/
		DATA qualifiers  (nqual_at17a+14)	/ 'DATASET'	/
		DATA qualifiers  (nqual_at17a+15)	/ '****'/

* SHADE - command # 18
	DATA commands(18)			/ 'SHADE'	/
        DATA its_action_command(18)             / .TRUE.        /
	DATA num_subcommands(18)		/ nsub18	/
	DATA subcommand_pointer(18)		/ nsub_at18	/
* SHADE '    '
	   DATA subcommands      (nsub_at18 + 0)	/ '    '	/
	   DATA num_qualifs      (nsub_at18 + 0)	/ nqual18a	/
	   DATA qualifier_pointer(nsub_at18 + 0)	/ nqual_at18a	/
* SHADE/I/J/K/L/M/N/X/Y/Z/T/E/F/DATASET/OVERLAY/SET_UP/FRAME/TRANSPOSE/TRIM
		DATA qualifiers  (nqual_at18a+0)/ 'I'	/
		DATA qualifiers  (nqual_at18a+1)/ 'J'	/
		DATA qualifiers  (nqual_at18a+2)/ 'K'	/
		DATA qualifiers  (nqual_at18a+3)/ 'L'	/
		DATA qualifiers  (nqual_at18a+4)/ 'M'	/
		DATA qualifiers  (nqual_at18a+5)/ 'N'	/
		DATA qualifiers  (nqual_at18a+6)/ 'X'	/
		DATA qualifiers  (nqual_at18a+7)/ 'Y'	/
		DATA qualifiers  (nqual_at18a+8)/ 'Z'	/
		DATA qualifiers  (nqual_at18a+9)/ 'T'	/
		DATA qualifiers  (nqual_at18a+10)/ 'E'	/
		DATA qualifiers  (nqual_at18a+11)/ 'F'	/
		DATA qualifiers  (nqual_at18a+12)/ 'OVERLAY'/	!/overlays
		DATA qualifiers  (nqual_at18a+13)/ 'SET_UP'/	!/set_up
		DATA qualifiers  (nqual_at18a+14)/ 'FRAME'/	!/frame
		DATA qualifiers  (nqual_at18a+15)/ 'DATASET'   /	!/data_set
		DATA qualifiers  (nqual_at18a+16)/ 'TRANPOSE'/	!/transpose
		DATA qualifiers  (nqual_at18a+17)/ 'TRIM'/	!/trim
		DATA qualifiers  (nqual_at18a+18)/ 'LINE'/	!/lines
		DATA qualifiers  (nqual_at18a+19)/ 'NOLABEL'/	!/nolabel
		DATA qualifiers  (nqual_at18a+20)/ 'LEVELS'/	!/levels_same
		DATA qualifiers  (nqual_at18a+21)/ 'KEY' /	!/key
		DATA qualifiers  (nqual_at18a+22)/ 'NOKEY'/	!/nokey
		DATA qualifiers  (nqual_at18a+23)/ 'PALETTE'/	!/palette=
		DATA qualifiers  (nqual_at18a+24)/ 'XLIMITS'/	!/xlimits
		DATA qualifiers  (nqual_at18a+25)/ 'YLIMITS'/	!/ylimits
		DATA qualifiers  (nqual_at18a+26)/ 'TITLE'/	!/title
		DATA qualifiers  (nqual_at18a+27)/ 'AXES'/	!/axes[=]
		DATA qualifiers  (nqual_at18a+28)/ 'NOAXES'/	!/noaxes
		DATA qualifiers  (nqual_at18a+29)/ 'PATTERN'/   !/pattern=
		DATA qualifiers  (nqual_at18a+30)/ 'HGRATICU'/!tic lines, dash or line
		DATA qualifiers  (nqual_at18a+31)/ 'VGRATICU'/!vert tic lines
		DATA qualifiers  (nqual_at18a+32)/ 'GRATICUL'/	!
		DATA qualifiers  (nqual_at18a+33)/ 'MODULO'/	! modulo for curvilinear plots
		DATA qualifiers  (nqual_at18a+34)/ 'HLIMITS'/	! horiz
		DATA qualifiers  (nqual_at18a+35)/ 'VLIMITS'/	! vertical
		DATA qualifiers  (nqual_at18a+36)/ 'OPACITY'/	! overriding opacity
		DATA qualifiers  (nqual_at18a+37)/ 'DEGMINSEC'/	! Request deg/min or deg/min/sec axis labels
		DATA qualifiers  (nqual_at18a+38)/ 'HALFSPAC'/	! Request a space before the E/W or N/S on axis

* SPAWN - command # 19
	DATA commands(19)			/ 'SPAWN'	/
        DATA its_action_command(19)             / .FALSE.       /
	DATA num_subcommands(19)		/ nsub19	/
	DATA subcommand_pointer(19)		/ nsub_at19	/
* SPAWN '    '
	   DATA subcommands      (nsub_at19 + 0)	/ '    '	/
	   DATA num_qualifs      (nsub_at19 + 0)	/ nqual19a	/
	   DATA qualifier_pointer(nsub_at19 + 0)	/ nqual_at19a	/

* **** formerly USER - command # 20
	DATA commands(20)			/ '****'	/
        DATA its_action_command(20)             / .TRUE.        /
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
		DATA qualifiers  (nqual_at20a+11)/ 'DATASET'		/
		DATA qualifiers  (nqual_at20a+12)/ 'FILE'	/
		DATA qualifiers  (nqual_at20a+13)/ 'FORMAT'	/

* WIRE - command # 21
	DATA commands(21)			/ 'WIRE'	/
        DATA its_action_command(21)             / .TRUE.       /
	DATA num_subcommands(21)		/ nsub21	/
	DATA subcommand_pointer(21)		/ nsub_at21	/
* WIRE '    '
	   DATA subcommands      (nsub_at21 + 0)	/ '    '	/
	   DATA num_qualifs      (nsub_at21 + 0)	/ nqual21a	/
	   DATA qualifier_pointer(nsub_at21 + 0)	/ nqual_at21a	/
* WIRE/I/J/K/L/M/N/X/Y/Z/T/E/F/DATASET/OVERLAY/SET_UP/FRAME/TRANSPOSE
		DATA qualifiers  (nqual_at21a+0) / 'I'	/
		DATA qualifiers  (nqual_at21a+1) / 'J'	/
		DATA qualifiers  (nqual_at21a+2) / 'K'	/
		DATA qualifiers  (nqual_at21a+3) / 'L'	/
		DATA qualifiers  (nqual_at21a+4) / 'M'	/
		DATA qualifiers  (nqual_at21a+5) / 'N'	/
		DATA qualifiers  (nqual_at21a+6) / 'X'	/
		DATA qualifiers  (nqual_at21a+7) / 'Y'	/
		DATA qualifiers  (nqual_at21a+8) / 'Z'	/
		DATA qualifiers  (nqual_at21a+9) / 'T'	/
		DATA qualifiers  (nqual_at21a+10) / 'E'	/
		DATA qualifiers  (nqual_at21a+11) / 'F'	/
		DATA qualifiers  (nqual_at21a+12) / 'OVERLAY'/ 
		DATA qualifiers  (nqual_at21a+13) / 'SET_UP'/  
		DATA qualifiers  (nqual_at21a+14)/ 'FRAME'/
		DATA qualifiers  (nqual_at21a+15)/ 'DATASET'   /
		DATA qualifiers  (nqual_at21a+16)/ 'VIEWPOIN'/
		DATA qualifiers  (nqual_at21a+17)/ 'ZLIMITS'/
		DATA qualifiers  (nqual_at21a+18)/ 'TRANPOSE'/  
		DATA qualifiers  (nqual_at21a+19)/ 'NOLABEL'/   
		DATA qualifiers  (nqual_at21a+20)/ 'ZSCALE'/
		DATA qualifiers  (nqual_at21a+21)/ 'COLOR'/
		DATA qualifiers  (nqual_at21a+22)/ 'THICKNES'/
		DATA qualifiers  (nqual_at21a+23)/ 'OPACITY'/
		DATA qualifiers  (nqual_at21a+24)/ '****'/
		DATA qualifiers  (nqual_at21a+25)/ '****'/
		DATA qualifiers  (nqual_at21a+26)/ 'TITLE'/

* QUERY - command # 22
	DATA commands(22)			/ 'QUERY'	/
        DATA its_action_command(22)             / .FALSE.       /
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
        DATA its_action_command(23)             / .FALSE.       /
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
        DATA its_action_command(24)             / .FALSE.       /
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
        DATA its_action_command(25)             / .FALSE.       /
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
        DATA its_action_command(26)             / .FALSE.       /
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
        DATA its_action_command(27)             / .TRUE.        /
	DATA num_subcommands(27)		/ nsub6		/
	DATA subcommand_pointer(27)		/ nsub_at27	/
* POLYGON '    '
	   DATA subcommands      (nsub_at27 + 0)/ '    '	/
	   DATA num_qualifs      (nsub_at27 + 0)/ nqual27a	/
	   DATA qualifier_pointer(nsub_at27 + 0)/ nqual_at27a	/
* POLYGON/I/J/K/L/M/N/X/Y/Z/T/E/F/DATASET/OVERLAY/SET_UP/FRAME/TRANSPOSE/COORD_AX
		DATA qualifiers  (nqual_at27a+0)/ 'I'	/
		DATA qualifiers  (nqual_at27a+1)/ 'J'	/
		DATA qualifiers  (nqual_at27a+2)/ 'K'	/
		DATA qualifiers  (nqual_at27a+3)/ 'L'	/
		DATA qualifiers  (nqual_at27a+4)/ 'M'	/
		DATA qualifiers  (nqual_at27a+5)/ 'N'	/
		DATA qualifiers  (nqual_at27a+6)/ 'X'	/
		DATA qualifiers  (nqual_at27a+7)/ 'Y'	/
		DATA qualifiers  (nqual_at27a+8)/ 'Z'	/
		DATA qualifiers  (nqual_at27a+9)/ 'T'	/
		DATA qualifiers  (nqual_at27a+10)/ 'E'	/
		DATA qualifiers  (nqual_at27a+11)/ 'F'	/
		DATA qualifiers  (nqual_at27a+12)/ 'OVERLAY'/	!/overlays
		DATA qualifiers  (nqual_at27a+13)/ 'SET_UP'/	!/set_up
		DATA qualifiers  (nqual_at27a+14)/ 'FRAME'/	!/frame
		DATA qualifiers  (nqual_at27a+15)/ 'DATASET' /	!/data_set
		DATA qualifiers  (nqual_at27a+16)/ 'TRANPOSE'/	!/transpose
		DATA qualifiers  (nqual_at27a+17)/ 'COORD_AX'/	!/coord_ax
		DATA qualifiers  (nqual_at27a+18)/ '****'/	!/unused
		DATA qualifiers  (nqual_at27a+19)/ 'NOLABEL'/	!/nolabel
		DATA qualifiers  (nqual_at27a+20)/ 'LEVELS'/	!/levels_same
		DATA qualifiers  (nqual_at27a+21)/ 'LINE'/	!/line
		DATA qualifiers  (nqual_at27a+22)/ 'COLOR'/	!/color
		DATA qualifiers  (nqual_at27a+23)/ 'PALETTE'/	!/palette=
		DATA qualifiers  (nqual_at27a+24)/ 'XLIMITS'/	!/xlimits
		DATA qualifiers  (nqual_at27a+25)/ 'YLIMITS'/	!/ylimits
		DATA qualifiers  (nqual_at27a+26)/ 'TITLE'/	!/title
		DATA qualifiers  (nqual_at27a+27)/ 'THICKNES'/	!/thickness
		DATA qualifiers  (nqual_at27a+28)/ 'NOAXES'/	!/noaxes
	        DATA qualifiers  (nqual_at27a+29)/ 'PATTERN'/   !/pattern=
		DATA qualifiers  (nqual_at27a+30)/ 'FILL'/	!/fill
		DATA qualifiers  (nqual_at27a+31)/ 'KEY'/	!/key
		DATA qualifiers  (nqual_at27a+32)/ 'NOKEY'/	!/nokey
		DATA qualifiers  (nqual_at27a+33)/ 'AXES'/	!/axes[=]
		DATA qualifiers  (nqual_at27a+34)/ 'HLIMITS'/	! horizontal
		DATA qualifiers  (nqual_at27a+35)/ 'VLIMITS'/	! vertical
		DATA qualifiers  (nqual_at27a+36)/ 'HLOG'/	!log axis horiz
		DATA qualifiers  (nqual_at27a+37)/ 'VLOG'/	!log axis vert
		DATA qualifiers  (nqual_at27a+38)/ 'HGRATICU'/	!tic lines, dash or line
		DATA qualifiers  (nqual_at27a+39)/ 'VGRATICU'/	!vert tic lines
		DATA qualifiers  (nqual_at27a+40)/ 'GRATICUL'/
		DATA qualifiers  (nqual_at27a+41)/ 'MODULO'/
		DATA qualifiers  (nqual_at27a+42)/ 'OPACITY'/	! overriding opacity
		DATA qualifiers  (nqual_at27a+43)/ '****'/

* ANNOTATE - command # 28
	DATA commands(28)			/ 'ANNOTATE'	/
        DATA its_action_command(28)             / .FALSE.       /
	DATA num_subcommands(28)		/ nsub28	/
	DATA subcommand_pointer(28)		/ nsub_at28	/
* ANNOTATE '    '
	   DATA subcommands      (nsub_at28 + 0)	/ '    '	/
	   DATA num_qualifs      (nsub_at28 + 0)	/ nqual28a	/
	   DATA qualifier_pointer(nsub_at28 + 0)	/ nqual_at28a	/
* ANNOTATE /USER/NORM/NOUSER/XPOS/YPOS/HALIGN/VALIGN/ANGLE/SIZE
		DATA qualifiers  (nqual_at28a+0)/ 'USER'	/
		DATA qualifiers  (nqual_at28a+1)/ 'NORM'	/
		DATA qualifiers  (nqual_at28a+2)/ 'NOUSER'	/
		DATA qualifiers  (nqual_at28a+3)/ 'XPOS'	/
		DATA qualifiers  (nqual_at28a+4)/ 'YPOS'	/
		DATA qualifiers  (nqual_at28a+5)/ 'HALIGN'	/
		DATA qualifiers  (nqual_at28a+6)/ 'VALIGN'	/
		DATA qualifiers  (nqual_at28a+7)/ 'ANGLE'	/
		DATA qualifiers  (nqual_at28a+8)/ 'SIZE'	/
		DATA qualifiers  (nqual_at28a+9)/ '****'	/
		DATA qualifiers  (nqual_at28a+10)/ '****'	/
		DATA qualifiers  (nqual_at28a+11)/ '****'	/

* **** - command # 29
	DATA commands(29)			/ '****'	/
        DATA its_action_command(29)             / .FALSE.       /
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
        DATA its_action_command(30)             / .FALSE.       /
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
     .		alias_len(13)	/ 14 /,
     .		alias(13)	/ 'REPEAT/ANIMATE' /,
     .		alias_name(14)	/ 'PATTERN' /,
     .		alias_len(14)	/ 19 /,
     .		alias(14)	/ 'PPL PATSET PATTERN=' /
     .		alias_name(15)	/ 'PAUSE' /,
     .		alias_len(15)	/ 7 /,
     .		alias(15)	/ 'MESSAGE' /,
     .		alias_name(16)	/ 'WHERE' /,
     .		alias_len(16)	/ 10 /,
     .		alias(16)	/ 'PPL %WHERE' /
     .		alias_name(17)	/ 'FLOWLINE' /,
     .		alias_len(17)	/ 17 /,
     .		alias(17)	/ 'VECTOR/FLOWLINE' /
     .		alias_name(18)	/ 'COLUMNS' /,
     .		alias_len(18)	/ 23 /,
     .		alias(18)	/ 'SET DATA/FORM=delimited' /
     .		alias_name(19)	/ 'KEYMARK' /,
     .		alias_len(19)	/ 23 /,
     .		alias(19)	/ 'PPL SHAKEY 1,,,,,,,,,,' /
     .		alias_name(20)	/ 'RIBBON' /,
     .		alias_len(20)	/ 11 /,
     .		alias(20)	/ 'PLOT/RIBBON' /
     .		alias_name(21)	/ 'ENSEMBLE' /,
     .		alias_len(21)	/ 22 /,
     .		alias(21)	/ 'DEFINE DATA/AGGREGAT/E' /
     .		alias_name(22)	/ 'FMRC' /,
     .		alias_len(22)	/ 22 /,
     .		alias(22)	/ 'DEFINE DATA/AGGREGAT/F' /
     .		alias_name(23)	/ 'VTREE' /,
     .		alias_len(23)	/ 13 /,
     .		alias(23)	/ 'SHOW VAR/TREE' /,
     .		alias_name(24)	/ 'TSERIES' /,
     .		alias_len(24)	/ 22 /,
     .		alias(24)	/ 'DEFINE DATA/AGGREGAT/T' /
     .		alias_name(25)	/ 'UNION' /,
     .		alias_len(25)	/ 22 /,
     .		alias(25)	/ 'DEFINE DATA/AGGREGAT/U' /
     .		alias_name(26)	/ 'FMASK' /,
     .		alias_len(26)	/ 14 /,
     .		alias(26)	/ 'SET DATA/FMASK' /
     .		alias_name(27)	/ 'SMASK' /,
     .		alias_len(27)	/ 17 /,
     .		alias(27)	/ 'SET DATA/SMASK' /
     .		alias_name(28)	/ 'CLEARMASK' /,
     .		alias_len(28)	/ 17 /,
     .		alias(28)	/ 'CANCEL DATA/MASK' /

        DATA
     .		(alias_name(i),i=num_predefined_alias+1,150)
     .                   / 122*unspecified_name4 /,
     .		(alias_len(i),i=num_predefined_alias+1,150) / 122*4 /,
     .		(alias(i),i=num_predefined_alias+1,150)
     .                   / 122*unspecified_name4 /

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

