/*
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
*/

/*****************************************************************************
 *                                                                           *
 * gksm2ps - translate GKS Metafiles to PostScript                           *
 *                                                                           *
 * Programmer - Larry Oolman                                                 *
 *              Department of Atmospheric Science                            *
 *              University of Wyoming                                        *
 *              oolman@coyote.uwyo.edu                                       *
 *                                                                           *
 * Date - 20 August 1991                                                     *
 *                                                                           *
 * *jd* 9.93      Mods to clean up line types, geometry handling, etc        *
 * *jd* 10.27.93  Mod: break on item type 4; xpplp now free of private items *
 * *jd* 11.17.93  Mod: handle items 71, 72 to generate default plot scaling  *
 * *jd* 11.29.93  Mod: add absolute_scaling option to size plot as original  *
 * *jd* 12.03.93  Mod: add color handling for fill area and colored lines    *
 * *jd* 12.06.93  Mod: setup so that multiple files are handled properly     *
 * *jd* 03.04.94  Mod: change argument syntax to match mtt                   *
 * *jd* 04.14.94  Mod: to handle version number of metafile                  *
 * *jd* 01.20.95  Mod: Now EPS compliant for single metafile translations    *
 * *jd* 09.15.95  Mod: Permit X window preview                               *
 * *jd* 12.23.97  Mod: Handle polyline rep setting for monochrome devices    *
 * *jd* 10.12.98  Mod: Support fill patterns                                 *
 *                                                                           *
 * *kob* 11.02    Mod: Support CMYK output
 *****************************************************************************/

/*  
 *  Mod 1.00 5.94       with release of xgks ferret
 *  Mod 1.01 6.28.94    fix bug in xoff, yoff when -geometry option used
 *  Mod 1.02 1.20.95    emits encapsulated PS when 1 file is translated
 *  Mod 1.03 6.14.95    Fix BB bug, clipping bug
 *  Mod 1.04 9.21.95    Add -X preview option
 *  Mod 1.05 12.23.97   Handle polyline rep setting for monochrome devices
 *  Mod 1.06 11.02.98   Support fill patterns
 */


/* Set these three items to the desired default values */

#define PAGE_WIDTH  10 	/* Default page width in inches */
#define PAGE_HEIGHT 10  /* Default page height in inches */
#define ORIENTATION 1   /* Default page orientation (0=portrait,1=landscape) */

#include "udposix.h"
#include <math.h>
#include <X11/Xlib.h>
#include <X11/Xutil.h>
#include <xgks.h>
#include <gks_implem.h>
#include <stdio.h>

#include <time.h>

unsigned int width  = 72*PAGE_WIDTH;    /* width of the plot (72 points = 1 inch) */
unsigned int height = 72*PAGE_HEIGHT;	/* height of the plot (72 points = 1 inch) */
int	 landscape  = ORIENTATION;	/* set for landscape mode (0=portrait, 1=landscape) */

int	individual = 1;			/* set if plots should be separated instead of overlayed */
int	written = 0;			/* set if anything has been written to the page */
int	xoff = 0;			/* xoffset (points) */
int	yoff = 0;			/* yoffset (points) */

int	xstat;          		/* status from an X function */

int     user_set_geometry = 0;          /* 1 if user uses -geometry option when invoking */
int     user_set_orientation = 0;       /* 1 if user uses landscape/portrait options when invoking */
int     scaling_completed = 0;          /* 1 when plot has been scaled to page */
int     absolute_scale = 0;             /* 1 when wsvp size plot is desired */
int     color_lines = 0;                /* 1 when color is used for lines */
int     rename_file = 1;                /* 1 if input files will be renamed */
int     use_cmyk = 0;                   /* 1 if PS output should be CMYK rather than RGB  *kob* 11/02 */

int	xborder = 18;			/* border on paper */
int	yborder = 18;
int     itmp;

int	pgwidth  = 612;			/* 8.5" page width (points) */
int	pgheight = 792;			/* 11" page height (points) */

int marksizemult = 4;

enum    wktype {cps, phaser};           /* Output graphics device */
enum    wktype device = cps;

FILE    *ps_output;                     /* PS output file */
char    output_file[80];                /* PS output file name */

int	nfile;		                /* number of input metafiles */
int     BB[4];                          /* Bounding Box information */

Gasfs	asf_list;			/* aspect souce flag list */
int	line_index = 1;			/* bundled polyline index */

Glnbundl line_individual =		/* individual line attributes */
	{ 1, 1.0, 1 };

Glnbundl line_bundle[MAX_BUNDL_TBL];

Glnbundl bw_line_bundle[MAX_BUNDL_TBL] = { /* monochrome line bundle table */
	{ 1, 1.0, 1 },
	{ 1, 1.0, 1 },
        { 2, 1.0, 1 },
	{ 3, 1.0, 1 },
	{ 4, 1.0, 1 },
	{ 5, 1.0, 1 },
	{ 6, 1.0, 1 },
	{ 1, 3.0, 1 },
        { 2, 3.0, 1 },
	{ 3, 3.0, 1 },
	{ 4, 3.0, 1 },
	{ 5, 3.0, 1 },
	{ 6, 3.0, 1 },
	{ 1, 5.0, 1 },
        { 2, 5.0, 1 },
	{ 3, 5.0, 1 },
	{ 4, 5.0, 1 },
	{ 5, 5.0, 1 },
	{ 6, 5.0, 1 },
	{ 1, 1.0, 0 } };

Glnbundl color_line_bundle[MAX_BUNDL_TBL] = { /* colored line bundle table */
	{ 1, 1.0, 1 },
	{ 1, 1.0, 1 },
        { 1, 1.0, 2 },
	{ 1, 1.0, 3 },
	{ 1, 1.0, 4 },
	{ 1, 1.0, 5 },
	{ 1, 1.0, 6 },
	{ 1, 3.0, 1 },
        { 1, 3.0, 2 },
	{ 1, 3.0, 3 },
	{ 1, 3.0, 4 },
	{ 1, 3.0, 5 },
	{ 1, 3.0, 6 },
	{ 1, 5.0, 1 },
        { 1, 5.0, 2 },
	{ 1, 5.0, 3 },
	{ 1, 5.0, 4 },
	{ 1, 5.0, 5 },
	{ 1, 5.0, 6 },
	{ 1, 1.0, 1 } };

int	mark_index = 1;			/* bundled polymarker index */

Gmkbundl mark_individual =		/* individual marker attributes */
	{ 3, 6.0, 1 };

Gmkbundl mark_bundle[MAX_BUNDL_TBL] = {	/* marker bundle table */
	{ 3, 6.0, 1 },
        { 1, 6.0, 1 },
	{ 2, 6.0, 1 },
	{ 4, 6.0, 1 },
	{ 5, 6.0, 1 } };

Gflbundl fill_individual =		/* individual fill attributes */
	{ GHOLLOW,  1, 1 };

Gflbundl fill_bundle[MAX_BUNDL_TBL] = {	/* bundled fill attributes */
	{ GHOLLOW,  1, 1 },
	{ GSOLID,   1, 1 },
	{ GPATTERN, 1, 1 },
	{ GHATCH,  -1, 1 },
	{ GHATCH, -10, 1 } };

int	fill_index = 1;			/* bundled fill index */
char	*fill = "stroke";		/* hollow or filled */

char *dash[] = { "[]",		        /* dash patterns */ 
          	 "[.007]",
		 "[.001 .002]",
		 "[.007 .003 .001 .003]",
		 "[.010 .004]",
          	 "[.006 .003 .001 .003 .001 .003]",
          	 "[.020 .020 .020 .020]",
		 "[.020 .020 .020 .020]",
		 "[.020 .020 .020 .020]",
		 "[.020 .020 .020 .020]",
		 "[.020 .020 .020 .020]" };
 
/*
 * char *dash[] = { "[]",			*** BIG dash patterns ***
 *         	 "[.007 .005]",
 *		 "[.002 .005]",
 *		 "[.014 .006 .002 .006]",
 *		 "[.025 .008]",
 *          	 "[.012 .006 .002 .006 .002 .006]",
 *         	 "[.020 .020 .020 .020]",
 *		 "[.020 .020 .020 .020]",
 *		 "[.020 .020 .020 .020]",
 *		 "[.020 .020 .020 .020]",
 *		 "[.020 .020 .020 .020]" };
 */

initpage ()		/* Initializes PostScript */
{
   char s[100];
   time_t now;
   now = time(NULL);
   strftime(s, 100, "%H:%M:%S %A, %d %B %Y", localtime(&now));

   fprintf(ps_output, "%%!PS-Adobe-1.0\n" );
   fprintf(ps_output, 
	   "%%%%Title: %s\n", output_file);
   fprintf(ps_output, 
	   "%%%%Creator: gksm2ps Mod 1.06 / XPPLP Profile F 1.0 [oX]\n" );
   fprintf(ps_output, 
	   "%%%%CreationDate: %s\n", s);


   if (nfile == 1) fprintf(ps_output, "%%%%BoundingBox:  %d %d %d %d\n", 
	   BB[0], BB[1], BB[2], BB[3] );

   fprintf(ps_output, "\n" );
   fprintf(ps_output, "%% define macros\n" );

   /* Pattern support macros */
   fprintf(ps_output, " \n");
   fprintf(ps_output, "%% Pattern support \n" );
   fprintf(ps_output, " \n");
   fprintf(ps_output, "/pixels { \n");
   fprintf(ps_output, "	/size exch def \n");
   fprintf(ps_output, "	/pattern exch def \n");
   fprintf(ps_output, "	 \n");
   fprintf(ps_output, "	size size  \n");
   fprintf(ps_output, "	true \n");
   fprintf(ps_output, "	[size 0 0 size 0 0] \n");
   fprintf(ps_output, "	pattern \n");
   fprintf(ps_output, "	imagemask \n");
   fprintf(ps_output, "} def \n");
   fprintf(ps_output, " \n");
   fprintf(ps_output, "/showpattern { \n");
   fprintf(ps_output, "	/ymul exch def \n");
   fprintf(ps_output, "	/xmul exch def \n");
   fprintf(ps_output, "	/size exch def \n");
   fprintf(ps_output, "	/pattern exch def  \n");
   fprintf(ps_output, " \n");
   fprintf(ps_output, "	xmul { ymul { pattern size pixels \n");
   fprintf(ps_output, " 		      0 1 translate \n");
   fprintf(ps_output, "	       	    } repeat \n");
   fprintf(ps_output, " \n");
   fprintf(ps_output, "       	     1 ymul -1 mul translate \n");
   fprintf(ps_output, "             } repeat \n");
   fprintf(ps_output, "} def \n");
   fprintf(ps_output, " \n");
   fprintf(ps_output, "/pfill { \n");
   fprintf(ps_output, "	/yorigin exch def \n");
   fprintf(ps_output, "	/xorigin exch def \n");
   fprintf(ps_output, "	/ymul exch def \n");
   fprintf(ps_output, "	/xmul exch def \n");
   fprintf(ps_output, "	/iscale exch def \n");
   fprintf(ps_output, "	/size exch def \n");
   fprintf(ps_output, "	/pattern exch def \n");
   fprintf(ps_output, " \n");
   fprintf(ps_output, "	gsave \n");
   fprintf(ps_output, "	clip \n");
   fprintf(ps_output, "	xorigin yorigin translate \n");
   fprintf(ps_output, "	iscale iscale scale \n");
   fprintf(ps_output, "	pattern size xmul ymul showpattern \n");
   fprintf(ps_output, "	grestore \n");
   fprintf(ps_output, "} def \n");
   fprintf(ps_output, " \n");
   fprintf(ps_output, "%% Example usage... \n");
   fprintf(ps_output, "%% 1 o \n");
   fprintf(ps_output, "%% np \n");
   fprintf(ps_output, "%% 0.405230 0.352940 mv \n");
   fprintf(ps_output, "%% 0.405230 0.392160 ln \n");
   fprintf(ps_output, "%% 0.614380 0.392160 ln \n");
   fprintf(ps_output, "%% 0.614380 0.352940 ln \n");
   fprintf(ps_output, "%% cp  \n");
   fprintf(ps_output, "%% {<8844221188442211>} 8 .01 22 5 .40 .35 pfill \n");
   fprintf(ps_output, " \n");

   /* End of Pattern support macros */

   fprintf(ps_output, "/inch {72 mul} def\n" );
   fprintf(ps_output, "/ln {lineto} def\n" );
   fprintf(ps_output, "/rln {rlineto} def\n" );
   fprintf(ps_output, "/mv {moveto} def\n" );
   fprintf(ps_output, "/rmv {rmoveto} def\n" );
   fprintf(ps_output, "/np {newpath} def\n" );
   fprintf(ps_output, "/cp {closepath} def\n" );
   fprintf(ps_output, "/lw {3000 div setlinewidth} def\n" );
   if (use_cmyk) 
     fprintf(ps_output, "/o {ct exch get aload pop setrgbcolor currentcmykcolor setcmykcolor} def\n" );
   else 
     fprintf(ps_output, "/o {ct exch get aload pop setrgbcolor} def\n" );
   fprintf(ps_output, "\n" );
   fprintf(ps_output, "/rect {\n" );
   fprintf(ps_output, "   np\n" );
   fprintf(ps_output, "   3 -1 roll exch mv\n" );
   fprintf(ps_output, "   currentpoint 4 -1 roll exch ln\n" );
   fprintf(ps_output, "   currentpoint pop 3 -1 roll ln\n" );
   fprintf(ps_output, "   currentpoint exch pop ln\n" );
   fprintf(ps_output, "   cp\n" );
   fprintf(ps_output, "} def\n" );
   fprintf(ps_output, "\n" );
   fprintf(ps_output, "/text {\n" );
   fprintf(ps_output, "   gsave\n" );
   fprintf(ps_output, "   translate\n" );
   fprintf(ps_output, "   charvec concat\n" );
   fprintf(ps_output, "   dup stringwidth pop ha mul va mv\n" );
   fprintf(ps_output, "   show\n" );
   fprintf(ps_output, "   grestore\n" );
   fprintf(ps_output, "} def\n" );
   fprintf(ps_output, "\n" );
   fprintf(ps_output, "%% create a font for markers\n" );
   fprintf(ps_output, "10 dict dup begin\n" );
   fprintf(ps_output, "\n" );
   fprintf(ps_output, "/FontType 3 def\n" );
   fprintf(ps_output, "/FontMatrix [.001 0 0 .001 0 0] def\n" );
   fprintf(ps_output, "/FontBBox [-450 -450 450 450] def\n" );
   fprintf(ps_output, "\n" );
   fprintf(ps_output, "/Encoding 256 array def\n" );
   fprintf(ps_output, "0 1 255 {Encoding exch /.notdef put} for\n" );
   fprintf(ps_output, "Encoding\n" );
   fprintf(ps_output, "  dup (1) 0 get /dot    put\n" );
   fprintf(ps_output, "  dup (2) 0 get /plus   put\n" );
   fprintf(ps_output, "  dup (3) 0 get /star   put\n" );
   fprintf(ps_output, "  dup (4) 0 get /circle put\n" );
   fprintf(ps_output, "      (5) 0 get /cross  put\n" );
   fprintf(ps_output, "\n" );
   fprintf(ps_output, "/Metrics 6 dict def\n" );
   fprintf(ps_output, "Metrics begin\n" );
   fprintf(ps_output, "  /.notdef  0 def\n" );
   fprintf(ps_output, "  /dot    900 def\n" );
   fprintf(ps_output, "  /plus   900 def\n" );
   fprintf(ps_output, "  /star   900 def\n" );
   fprintf(ps_output, "  /circle 900 def\n" );
   fprintf(ps_output, "  /cross  900 def\n" );
   fprintf(ps_output, "end\n" );
   fprintf(ps_output, "\n" );
   fprintf(ps_output, "/BBox 6 dict def\n" );
   fprintf(ps_output, "BBox begin\n" );
   fprintf(ps_output, "  /.notdef [   0    0   0   0] def\n" );
   fprintf(ps_output, "  /dot     [ -15  -15  15  15] def\n" );
   fprintf(ps_output, "  /plus    [-450 -450 450 450] def\n" );
   fprintf(ps_output, "  /star    [-450 -450 450 450] def\n" );
   fprintf(ps_output, "  /circle  [-450 -450 450 450] def\n" );
   fprintf(ps_output, "  /cross   [-450 -450 450 450] def\n" );
   fprintf(ps_output, "end\n" );
   fprintf(ps_output, "\n" );
   fprintf(ps_output, "/CharacterDefs 6 dict def\n" );
   fprintf(ps_output, "CharacterDefs begin\n" );
   fprintf(ps_output, "  /.notdef {} def\n" );
   fprintf(ps_output, "  /dot     { np 0 0 15 0 360 arc cp fill } def\n" );
   fprintf(ps_output, "  /plus    { np 0 450 mv 0 -450 ln\n" );
   fprintf(ps_output, "		-450 0 mv 450 0 ln\n" );
   fprintf(ps_output, "		30 setlinewidth stroke } def\n" );
   fprintf(ps_output, "  /star    { np 0 450 mv 0 -450 ln\n" );
   fprintf(ps_output, "		-390 225 mv 390 -225 ln\n" );
   fprintf(ps_output, "		-390 -225 mv 390 225 ln\n" );
   fprintf(ps_output, "		30 setlinewidth stroke } def\n" );
   fprintf(ps_output, "  /circle  { np 0 0 435 0 360 arc cp \n" );
   fprintf(ps_output, "		30 setlinewidth stroke } def\n" );
   fprintf(ps_output, "  /cross   { np -318 318 mv 318 -318 ln\n" );
   fprintf(ps_output, "		-318 -318 mv 318 318 ln\n" );
   fprintf(ps_output, "		30 setlinewidth stroke } def\n" );
   fprintf(ps_output, "end\n" );
   fprintf(ps_output, "\n" );
   fprintf(ps_output, "/BuildChar\n" );
   fprintf(ps_output, "  { 0 begin\n" );
   fprintf(ps_output, "      /char exch def\n" );
   fprintf(ps_output, "      /fontdict exch def\n" );
   fprintf(ps_output, "      /charname fontdict /Encoding get char get def\n" );
   fprintf(ps_output, "      fontdict begin\n" );
   fprintf(ps_output, "        Metrics charname get 0\n" );
   fprintf(ps_output, "        BBox charname get aload pop\n" );
   fprintf(ps_output, "        setcachedevice\n" );
   fprintf(ps_output, "        CharacterDefs charname get exec\n" );
   fprintf(ps_output, "      end\n" );
   fprintf(ps_output, "    end\n" );
   fprintf(ps_output, "  } def\n" );
   fprintf(ps_output, "/BuildChar load 0 3 dict put\n" );
   fprintf(ps_output, "/UniqueID 1 def\n" );
   fprintf(ps_output, "end\n" );
   fprintf(ps_output, "\n" );
   fprintf(ps_output, "/MarkFont exch definefont pop\n" );
   fprintf(ps_output, "  \n" );
   fprintf(ps_output, "/mkinit { gsave /MarkFont findfont msize 72 div scalefont setfont } def\n" );
   fprintf(ps_output, "/marker { mv mtype show } def\n" );
   fprintf(ps_output, "\n" );
   fprintf(ps_output, "\n" );
   fprintf(ps_output, "%% initialize variables\n" );
   fprintf(ps_output, "/charvec {[.01 0 0 .01 0 0]} def\n" );
   fprintf(ps_output, "/ha {0} def\n" );
   fprintf(ps_output, "/va {0} def\n" );
   fprintf(ps_output, "/msize { 1 } def\n" );
   fprintf(ps_output, "/mtype { (1) } def\n" );
   fprintf(ps_output, "/ct 256 array def\n" );
   fprintf(ps_output, "\n" );
   fprintf(ps_output, "%% begin the plot\n" );
   fprintf(ps_output, "1 lw\n" );
   fprintf(ps_output, "/Courier findfont setfont\n\n" );
}

out_line_type( index )		/* output the line type */
   int index;
{
   fprintf(ps_output, "%s 0 setdash\n", dash[index-1] );
}

out_line_width( width )		/* output the line width */
   float width;
{
   fprintf(ps_output, "%f lw\n", width );
}

out_mark_type( index )		/* output the marker type */
   int index;
{
   fprintf(ps_output, "/mtype { (%d) } def\n", index );
}

out_mark_size( size )		/* output the marker size */
   float size;
{
   fprintf(ps_output, "/msize {%f} def\n", marksizemult*size );
}


find_BB( meta_id )	/* find the Bounding Box info */

/* Mod 1.20.95  BB info is determined by items of type 71 and 72.  This 
 * function hacked from ps_trans_meta to extract info from those two and
 * determine BB using Byzantine Balderdash algorithmick schema.
 *
 * Oops. Fix to make NON-square BB.  *jd* 6.13.95
 */

   int meta_id;
{
   Gint	   error;
   void    *record;
   Ggksmit gksmit;

   int          BBpgwidth, BBpgheight, BBxborder, BByborder, BBxoff, BByoff;
   unsigned int BBwidth, BBheight;

   int avail_pgheight;  /* available page height*/
   int avail_pgwidth;   /* available page width*/

   float   xndc,yndc;   /* ndc values for ws window (x,y) limits */
   int   xpoints,ypoints;   /* values for ws viewport (x,y) limits in points */

   float aspect;        /* aspect (x/y) ratio of plot */
   float pgaspect;      /* aspect (W/H) ratio of page */

   /* Initialize things */
   BBpgwidth  = pgwidth;
   BBpgheight = pgheight;
   BBxborder  = xborder;
   BByborder  = yborder;
   BBxoff     = xoff;
   BByoff     = yoff;
   BBwidth    = width;
   BBheight   = height;

   do {
      error = ggetgksm (meta_id, &gksmit);
      if (gksmit.type <= 0)		/* GKSM end record */
         break;
      record = malloc (gksmit.length);
      if ( (error = greadgksm(meta_id, gksmit.length, record)) != 0 )
	return; 

      switch (gksmit.type) {

         case 71:                       /* workstation window */
            xndc = (*(XGKSMLIMIT *)record).rect.xmax;
            yndc = (*(XGKSMLIMIT *)record).rect.ymax;
            aspect = xndc/yndc;

            if (!user_set_orientation) 
              landscape = (aspect >= 1.0) ? 1:0;

            if (landscape) {
              BBpgwidth  = pgheight;
              BBpgheight = pgwidth;

              BBxborder  = yborder;
              BByborder  = xborder;
            }
            break;

	 case 72:			/* workstation viewport */
	    avail_pgwidth  = BBpgwidth  - 2*BBxborder;
	    avail_pgheight = BBpgheight - 2*BByborder;

	    if (!user_set_geometry && !absolute_scale) { /* do best fit */
	      pgaspect = ((float) avail_pgwidth)/((float) avail_pgheight);

	      if (aspect >= 1.0) { /* width scales the plot*/
	        BBwidth = BBheight = (aspect >= pgaspect) ?
	          avail_pgwidth:avail_pgheight*aspect;

		/* Capture aspect ratio in BB */
		BBwidth  *= xndc;
		BBheight *= yndc; 

	        BBxoff = (avail_pgwidth  - BBwidth)/2;
	        BByoff = (avail_pgheight - BBheight)/2;
	      }
	      else { /* height scales the plot*/
	        BBwidth = BBheight = (aspect <= pgaspect) ?
	          avail_pgheight:avail_pgwidth/aspect;

		/* Capture aspect ratio in BB */
		BBwidth  *= xndc;
		BBheight *= yndc; 

	        BBxoff = (avail_pgwidth  - BBwidth)/2;
	        BByoff = (avail_pgheight - BBheight)/2;	       
	      }	     
	    }

	    if (absolute_scale) { /* use size specified in metafile */
	       xpoints = (*(XGKSMLIMIT *)record).rect.xmax*72;
   	       ypoints = (*(XGKSMLIMIT *)record).rect.ymax*72;
	       	
	       BBwidth = BBheight = (aspect >= 1.0) ? xpoints:ypoints;
	       
	       /* Capture aspect ratio in BB */
	       BBwidth  *= xndc;
	       BBheight *= yndc; 

	       BBxoff = (avail_pgwidth  - xpoints)/2;
	       BByoff = (avail_pgheight - ypoints)/2;
	    }

	    BB[0] = BBxborder + BBxoff;
	    BB[1] = BByborder + BByoff;
	    BB[2] = BB[0] + BBwidth;
	    BB[3] = BB[1] + BBheight;

	    free(record);
	    return 0; /* Get out */

	 default:
	    break;

      }

      free(record);

   } while (gksmit.type > 0);
}

ps_trans_meta( meta_id )        /* translate the metafile */

   int meta_id;
{
   Gint	   error;
   void    *record;
   Ggksmit gksmit;

   int i;		/* loop counter */
   int index;		/* index */
   int first_clip=1;    /* first use of clip */

   int tmp;             /* temporary value */
   int avail_pgheight;  /* available page height*/
   int avail_pgwidth;   /* available page width*/

   float   xndc,yndc;   /* ndc values for ws window (x,y) limits */
   int   xpoints,ypoints;   /* values for ws viewport (x,y) limits in points */

   float aspect;        /* aspect (x/y) ratio of plot */
   float pgaspect;      /* aspect (W/H) ratio of page */

   XGKSMGRAPH *line;	/* a line */
   Gpoint *point;	/* a point */

   char char_vec[80];	/* character vector transformation matrix */

   static float ha[4] = { 0.0,  0.0, -0.5, -1.0 };		/* horizontal text alignments */
   static float va[6] = { 0.0, -1.0, -1.2, -0.5, 0.0, 0.2 };	/* vertical text alignments */

   /* Use with pattern support */
   float xxmin, yymin, xxmax, yymax, xxorg, yyorg, iscale; 
   int ixmul, iymul, psize;                        

   char * pattern[] = {"",
		       "113377ff113377ff",
		       "ff9999ffff9999ff",
		       "0066660000666600",
		       "cccc3333cccc3333",
       		       "ff000000ff000000",
		       "8888888888888888",
		       "00ffffff00ffffff",
		       "eeeeeeeeeeeeeeee",
		       "77bbddee77bbddee",
		       "8844221188442211",
		       "eeddbb77eeddbb77",
		       "1122448811224488",
       		       "ff80808080808080",
		       "8142241818244281",
		       "9090909090909090",
	               "ff0000ff00000000",
       "f1f1f1f13131eeee1f1f1f1f1313eeeef1f1f1f13131eeee1f1f1f1f1313eeee",
       "f8f87474222247478f8f171722227171f8f87474222247478f8f171722227171",
       "2828101010107c7c828201010101c7c72828101010107c7c828201010101c7c7",
       "ffff080808080808ffff808080808080ffff080808080808ffff808080808080"};

   /***********************************************************************/

   do {
      error = ggetgksm (meta_id, &gksmit);
      if (gksmit.type <= 0)		/* GKSM end record */
         break;
      record = malloc (gksmit.length);
      if ( (error = greadgksm(meta_id, gksmit.length, record)) != 0 )
	return; 

      switch (gksmit.type) {
	 case 1:			/* clear workstation */
/*
	    if (individual&&written)  {
		fprintf(ps_output, "showpage\n");
		initpage();
	    }
            written = 0;
*/
	    break;
	 case  4:
	    break;
	 case 11:			/* polyline */
	    if (color_lines)
      	       fprintf(ps_output, "%d o\n",line_bundle[line_index].colour);
	    else {
	      if (line_index == 19) 
		fprintf(ps_output, "1.0 1.0 1.0 setrgbcolor\n"); /* use white-out*/
	      else
	        fprintf(ps_output, "0.0 0.0 0.0 setrgbcolor\n"); /* use black */
	  
	    }
	    point = (*(XGKSMGRAPH *)record).pts;
	    fprintf(ps_output, "np\n");
	    for ( i=0; i < (*(XGKSMGRAPH *)record).num_pts; i++ ) {
	       fprintf(ps_output, "%f %f %s\n",
		  point->x,
		  point->y,
		  i ? "ln" : "mv" );
	       point++;
	    }
	    fprintf(ps_output, "stroke\n");
            written = 1;
	    break;
	 case 12:			/* polymarker */
	    point = (*(XGKSMGRAPH *)record).pts;
	    fprintf(ps_output, "mkinit\n");
	    for ( i=0; i < (*(XGKSMGRAPH *)record).num_pts; i++ ) {
	       fprintf(ps_output, "%f %f marker\n",
		  point->x,
		  point->y );
	       point++;
	    }
	    fprintf(ps_output, "grestore\n");
            written = 1;
	    break;
	 case 13:			/* text */
	    fprintf(ps_output, "(%s) %f %f text\n",
	       (*(XGKSMTEXT *)record).string,
	       (*(XGKSMTEXT *)record).location.x,
	       (*(XGKSMTEXT *)record).location.y );
            written = 1;
	    break; 
	 case 14:			/* fill area */

	    fprintf(ps_output, "%d o\n",fill_individual.colour);
	    point = (*(XGKSMGRAPH *)record).pts;
	    fprintf(ps_output, "np\n");

	    xxmin = 1.0;
	    xxmax = 0.0;
	    yymin = 1.0;
	    yymax = 0.0;

	    for ( i=0; i < (*(XGKSMGRAPH *)record).num_pts; i++ ) {
	       fprintf(ps_output, "%f %f %s\n",
		  point->x,
		  point->y,
		  i ? "ln" : "mv" );

	       if (xxmin > point->x) 
		 xxmin = point->x;
 
	       if (yymin > point->y) 
		 yymin = point->y;

	       if (xxmax < point->x) 
		 xxmax = point->x;

	       if (yymax < point->y) 
		 yymax = point->y;

	       point++;
	    }

	    /* Pattern support */
	    if (fill_individual.inter == GHATCH)
	    {
	      if (strlen (pattern[(-1)*fill_individual.style]) == 16)
	      {
		xxorg = ((float) ( (int) (xxmin * 100.0) )) / 100.0; 
		yyorg = ((float) ( (int) (yymin * 100.0) )) / 100.0;
		
		ixmul = (int) (100 * (xxmax - xxorg + .01));
		iymul = (int) (100 * (yymax - yyorg + .01));
		
		psize = 8;
		iscale = .01;
	      }
	      else  if (strlen (pattern[(-1)*fill_individual.style]) == 64)
	      {
		xxorg = ((float) ( (int) (xxmin * 50.0) )) / 50.0; 
		yyorg = ((float) ( (int) (yymin * 50.0) )) / 50.0;
		
		ixmul = (int) (50 * (xxmax - xxorg + .02));
		iymul = (int) (50 * (yymax - yyorg + .02));
		
		psize = 16;
		iscale = .02;
	      }
	      
	      fprintf(ps_output, "cp \n");
	      fprintf(ps_output, 
		      "{<%s>} %d %.2f %d %d %.2f %.2f pfill\n",
		      pattern[(-1)*fill_individual.style], psize, iscale,
		      ixmul, iymul, xxorg, yyorg);
	    }
	    else
	      fprintf(ps_output, "cp %s\n",fill);
	
            written = 1;
	    break;
	 case 21:	      	/* polyline index */
	    line_index = *(long *)record;
	    if (asf_list.ln_type == GBUNDLED)
	       out_line_type( line_bundle[line_index].type );
	    if (asf_list.ln_width == GBUNDLED)
	       out_line_width( line_bundle[line_index].width );
	    break;
	 case 22:			/* linetype */
	    line_individual.type = (*(XGKSMONE *)record).flag;
	    if ( line_individual.type < 1 || line_individual.type > sizeof(dash)/sizeof(char *) ) {
	       fprintf( stderr, 
		  "GKSM22: line type %d is undefined, using solid\n",line_individual.type);
	       line_individual.type = 1;
	    }
	    if (asf_list.ln_type == GINDIVIDUAL)
	       out_line_type( line_individual.type );
	    break;
	 case 23:			/* linewidth scale factor */
	    line_individual.width = (*(XGKSMSIZE *)record).size;
	    if (asf_list.ln_width == GINDIVIDUAL)
	       out_line_width( line_individual.width );
	    break;
	 case 24:			/* polyline colour index */
	    line_individual.colour = (*(XGKSMONE *)record).flag;
	    break;
	 case 25:			/* polymarker index */
	    mark_index = (*(XGKSMONE *)record).flag;
	    if (asf_list.mk_type == GBUNDLED)
	       out_mark_type( mark_bundle[mark_index].type );
	    if (asf_list.mk_size == GBUNDLED)
 	       out_mark_size( mark_bundle[mark_index].size );
	    break;
	 case 26:			/* marker type */
	    mark_individual.type = (*(XGKSMONE *)record).flag;
	    if ( mark_individual.type < 1 || mark_individual.type > 5 ) {
	       fprintf( stderr, 
		  "GKSM26: marker type %d is undefined, using stars",mark_individual.type);
	       mark_individual.type = 3;
	    }
	    if (asf_list.mk_type == GINDIVIDUAL)
	       out_mark_type( mark_individual.type );
            break;
	 case 27:			/* marker size scale factor */
	    mark_individual.size = (*(XGKSMSIZE *)record).size;
	    if (asf_list.mk_size == GINDIVIDUAL)
	       out_mark_size( mark_individual.size );
	    break;
	 case 28:			/* polymarker colour index */
	    mark_individual.colour = (*(XGKSMONE *)record).flag;
	    break;
	 case 29:			/* text index */
	 case 30:			/* text font and precision */
	 case 31:			/* character expansion factor */
	 case 32:			/* character spacing */
	 case 33:			/* text colour index */
	    break;
	 case 34:			/* character vectors */
	    fprintf(ps_output, "/charvec {[%f %f %f %f 0 0]} def\n",
	       (*(XGKSMCHARVEC *)record).base.x,
	       (*(XGKSMCHARVEC *)record).base.y,
	       (*(XGKSMCHARVEC *)record).up.x,
	       (*(XGKSMCHARVEC *)record).up.y );
	    break;
	 case 35:			/* text path */
	    break;
	 case 36:			/* text alignment */
	    i = (*(XGKSMTWO *)record).item1 ;
	    if (i < 0 || i > 3) {
	       fprintf( stderr, "GKSM36: invalid ha: %d\n", i);
	       i = 0;
	    }
	    fprintf(ps_output, "/ha {%f} def\n",ha[i]);
	    i = (*(XGKSMTWO *)record).item2;
	    if (i < 0 || i > 5) {
	       fprintf( stderr, "GKSM36: invalid va: %d\n", i);
	       i = 0;
	    }
	    fprintf(ps_output, "/va {%f} def\n",va[i]);
	    break;
	 case 37:			/* fill area index */
	    fill_index = *(long *)record;
	    if (asf_list.fl_inter == GBUNDLED)
	       fill = fill_bundle[fill_index].inter==GHOLLOW ? "stroke" : "fill";
	    break;
	 case 38:			/* fill interior style */
	    fill_individual.inter = (*(XGKSMONE *)record).flag;
	    if (asf_list.fl_inter == GINDIVIDUAL)
	       fill = fill_bundle[fill_index].inter==GHOLLOW ? "stroke" : "fill";
	    break;
	 case 39:			/* fill area style index */
	    fill_individual.style = (*(XGKSMONE *)record).flag;
	    break;
	 case 40:			/* fill area colour index */
	    fill_individual.colour = (*(XGKSMONE *)record).flag;
	    break;
	 case 41:			/* pattern size */
	 case 42:			/* pattern reference point */
            break;
	 case 43:			/* aspect source flags */
	    asf_list = *(Gasfs *)record;
	    out_line_type ( asf_list.ln_type  == GBUNDLED ? line_bundle[line_index].type  : line_individual.type  );
	    out_line_width( asf_list.ln_width == GBUNDLED ? line_bundle[line_index].width : line_individual.width );
	    out_mark_type ( asf_list.mk_type  == GBUNDLED ? mark_bundle[mark_index].type  : mark_individual.type  );
	    out_mark_size ( asf_list.mk_size  == GBUNDLED ? mark_bundle[mark_index].size  : mark_individual.size  );
	    fill = (asf_list.fl_inter  == GBUNDLED ? fill_bundle[fill_index].inter : fill_individual.inter )==GHOLLOW ?
		"stroke" : "fill"; 
            break;
	 case 44:			/* pick identifier */
	    break;
         case 51:			/* polyline representation */
	    index = (*(XGKSMLMREP *)record).idx;
	    /* Ignore color and style setting if monochrome device */
	    if ( 0 < index < MAX_BUNDL_TBL ) {
	      if (color_lines) {
		line_bundle[index].type   = (*(XGKSMLMREP *)record).style;
		line_bundle[index].colour = (*(XGKSMLMREP *)record).color;
	      }
	      line_bundle[index].width  = (*(XGKSMLMREP *)record).size;
	    }
	    else
	      fprintf( stderr, "GKSM51: index not in range 0-%d\n", MAX_BUNDL_TBL-1 );
	    break;
	 case 52:			/* polymarker representation */
	    index = (*(XGKSMLMREP *)record).idx;
	    if ( 0 < index < MAX_BUNDL_TBL ) {
               mark_bundle[index].type   = (*(XGKSMLMREP *)record).style;
               mark_bundle[index].size   = (*(XGKSMLMREP *)record).size;
               mark_bundle[index].colour = (*(XGKSMLMREP *)record).color;
	    }
	    else
	       fprintf( stderr, "GKSM52: index not in range 0-%d\n", MAX_BUNDL_TBL-1 );
	    break;
	 case 54:			/* fill area representation */
	    index = (*(XGKSMFILLREP *)record).idx;
	    if ( 0 < index < MAX_BUNDL_TBL ) {
		fill_bundle[index].inter  = (*(XGKSMFILLREP *)record).intstyle;
		fill_bundle[index].style  = (*(XGKSMFILLREP *)record).style;
		fill_bundle[index].colour = (*(XGKSMFILLREP *)record).colour;
	    }
	    else
	       fprintf( stderr, "GKSM54: index not in range 0-%d\n", MAX_BUNDL_TBL-1 );
	    break;
	 case 56:			/* colour representation */
            fprintf(ps_output, "ct %d [%f %f %f] put\n",
	       (*(XGKSMCOLOURREP *)record).idx,
	       (*(XGKSMCOLOURREP *)record).red,
	       (*(XGKSMCOLOURREP *)record).green,
	       (*(XGKSMCOLOURREP *)record).blue);
	    break;
	 case 61:			/* clipping rectangle */
	    if (!scaling_completed) break;

	    if (first_clip)
	    {
	      first_clip = 0;
	      fprintf(ps_output, "0.0 1.0 0.0 1.0 rect clip\n"); 
	      fprintf(ps_output, "gsave\n");
	    }

	    /* Change initclip to grestore gsave -- initclip is non-EPS jd */
	    fprintf(ps_output, "grestore gsave %f %f %f %f rect clip\n", 
   	       (*(XGKSMLIMIT *)record).rect.xmin,
   	       (*(XGKSMLIMIT *)record).rect.xmax,
   	       (*(XGKSMLIMIT *)record).rect.ymin,
   	       (*(XGKSMLIMIT *)record).rect.ymax );
	    break;
	 case 71:			/* workstation window */
	    if (scaling_completed) break;
	    xndc = (*(XGKSMLIMIT *)record).rect.xmax;
   	    yndc = (*(XGKSMLIMIT *)record).rect.ymax;
	    aspect = xndc/yndc;

	    if (!user_set_orientation) 
	      landscape = (aspect >= 1.0) ? 1:0;

	    if (landscape) {
	      fprintf(ps_output, "-90 rotate -11 inch 0 inch translate\n" );
	      tmp      = pgwidth;
	      pgwidth  = pgheight;
	      pgheight = tmp;

	      itmp     = xborder;
	      xborder  = yborder;
	      yborder  = itmp;
	    }
	    break;
	 case 72:			/* workstation viewport */
	    if (scaling_completed) break;
	    avail_pgwidth  = pgwidth  - 2*xborder;
	    avail_pgheight = pgheight - 2*yborder;

	    if (!user_set_geometry && !absolute_scale) { /* do best fit */
	      pgaspect = ((float) avail_pgwidth)/((float) avail_pgheight);

	      if (aspect >= 1.0) { /* width scales the plot*/
	        width = height = (aspect >= pgaspect) ?
	          avail_pgwidth:avail_pgheight*aspect;

	        xoff = (avail_pgwidth  - width)/2;
	        yoff = (avail_pgheight - height/aspect)/2;
	      }
	      else { /* height scales the plot*/
	        width = height = (aspect <= pgaspect) ?
	          avail_pgheight:avail_pgwidth/aspect;

	        xoff = (avail_pgwidth  - width*aspect)/2;
	        yoff = (avail_pgheight - height)/2;	       
	      }
	    }

	    if (absolute_scale) { /* use size specified in metafile */
	       xpoints = (*(XGKSMLIMIT *)record).rect.xmax*72;
   	       ypoints = (*(XGKSMLIMIT *)record).rect.ymax*72;
	       
	       width = height = (aspect >= 1.0) ? xpoints:ypoints;
	       xoff = (avail_pgwidth  - xpoints)/2;
	       yoff = (avail_pgheight - ypoints)/2;
            }
	    fprintf(ps_output, "%d %d translate\n", xborder+xoff,yborder+yoff );
	    fprintf(ps_output, "%d %d scale\n", width, height );

	    scaling_completed = 1;
	    break;
	 case 81:			/* create segment */
	 case 82:			/* close  segment */
	 case 83:			/* rename segment */
	 case 84:			/* delete segment */
	    break;
	 default:
	    fprintf( stderr, "GKSM item %d is undefined\n", gksmit.type);
	    break;
      }

      free(record);

   } while (gksmit.type > 0);
}

void time_stamp (name, stamp)

char name[], stamp[];

/* Add time stamp to file name and rename file 
 *
 * *jd* 3.6.94                                 
 */

{ 
  int         lname, lext;
  char        *sptr, new_name[90], command[180];

  sptr  = (char *) strchr (name, '.');
  lname = strlen (name);
  lext  = (sptr != NULL) ? strlen (sptr) : 0;
 
  strcpy (new_name, name);
  new_name[lname - lext] = '\0';

  strcat (new_name, stamp);
  if (sptr != NULL) strcat (new_name, sptr);
 
  sprintf (command, "mv %s %s", name, new_name);
  system  (command);
}

#define MAXFILE 100

main (argc, argv)
   int     argc;
   char   *argv[];

{
   int     i;			/* loop counter */
   char    file[MAXFILE][80];	/* The file name */
   char    stamp[8];            /* time stamp */
   time_t  now;                 /* time gksm2ps called */

   int     work_id;		/* Workstation identifier */
   int     con_id;		/* Workstation connection identifier */
   char	   work_type[80];	/* Workstation type */
   FILE	   err_fil;		/* Error file */

   char	   *metatype = "MI";	/* Input metafile workstation type */
   int	   meta_id;		/* Metafile workstation identifier */
   int	   gks_stat;		/* The return status from gks function */
   char    *getenv();

   char    *mod = "1.06";       /* -X preview option added*/
   char	   *gksm2ps_version = "XPPLP Profile F 1.0 "; /* gksm2ps version # */
   char    metafile_header[45];   /* Version of metafiles read in */
   char    *metafile_version;

   FILE    *meta_file;           /* meta file */

   int     xpreview_set=0;       /* Use Xwindow preview? */
   char    answer[40];           /* Answer to Q's in X preview */  
   int ch;


/* clear array of file names */
   for (i=0; i<MAXFILE; i++)
      strcpy(file[i],"\0");
   nfile = 0;

/* Initialize ps output file name */
   strcpy (output_file, "gksm2ps_output.ps");

   for (i = 1; i < argc; i++) {
   /* Specifies help */
      if (strcmp(argv[i], "-H")==0 || strcmp(argv[i], "-h")==0 || strcmp(argv[i], "-?")==0 || strcmp(argv[i], "-help")==0) {
	 printf("gksm2ps: Send PostScript translation of GKSM metafiles to a file\n");
	 printf("  usage: gksm2ps [-h] [-p landscape||portrait] [-l ps||cps] [-d cps||phaser] \\\n");
	 printf("         [-X || -o <ps_output_file>] [-R] [-a] [-g WxH+X+Y] [-v] [-C] file(s)\n\n");	
	 printf("     -h: print this help message\n");
	 printf("     -p: page orientation, landscape or portrait (default fits to page)\n");
	 printf("     -l: line styles,  ps == monochrome (default), cps == color\n");
	 printf("     -d: device type, cps == Postscript (default), phaser == TEK phaser PS\n");
	 printf("     -X: Send output to your Xwindow for preview instead of a file\n");
	 printf("     -o: PS output file name (default name is 'gksm2ps_output.ps')\n");
	 printf("     -R: do not rename files with a date stamp appended (default is to stamp)\n");	 
	 printf("     -a: make hard copy the size of the original plot (default fits to page)\n");
	 printf("     -g: WxH+X+Y  WIDTH, HEIGHT, XOFFSET, & YOFFSET in points (72 pts = 1 in)\n");
	 printf("     -v: list version number of gksm2ps and do nothing else\n");
	 printf("     -C: Output a CMYK postscript file (default is RGB)\n\n");

	 printf("file(s): The specific metafile(s) to be translated.\n\n");
	 exit(0);
      }

      else if ( !strcmp(argv[i], "-geometry") || !strcmp(argv[i], "-g") ) {
	 xstat =XParseGeometry(argv[++i], &xoff, &yoff, &width, &height);
	 user_set_geometry = 1;
      }
	 
      else if ( !strcmp(argv[i], "-absolute_scale") || !strcmp(argv[i], "-a") ) {
         absolute_scale = 1;
      }

      else if ( !strcmp(argv[i], "-page") || !strcmp(argv[i], "-p") ) {
         i++;
         if      ( !strcmp(argv[i], "landscape")) landscape = 1;
         else if ( !strcmp(argv[i], "portrait"))  landscape = 0;
         else { fprintf( stderr, "%s is not a valid page orientation\n", argv[i]);
            exit(1);
         }
	 user_set_orientation = 1;
      }
	 
      else if ( !strcmp(argv[i], "-line")  || !strcmp(argv[i], "-l")) {
         i++;
         if      ( !strcmp(argv[i], "cps")) color_lines = 1;
         else if ( !strcmp(argv[i], "ps"))  color_lines = 0;
         else { fprintf( stderr, "%s is not a valid line color specification\n", argv[i]);
            exit(1);
         }
      }

      /* Permit cps and phaser wstypes */
      else if ( !strcmp(argv[i], "-dev") || !strcmp(argv[i], "-d") ) {
         i++;
         if      ( !strcmp(argv[i], "cps"))     device = cps;
         else if ( !strcmp(argv[i], "phaser"))  device = phaser;
         else { fprintf( stderr, "%s is not a valid device type\n", argv[i]);
            exit(1);
         }
      }	 
 
      else if ( !strcmp(argv[i], "-Xwindow") || !strcmp(argv[i], "-X") ) 
	xpreview_set = 1;

      else if ( !strcmp(argv[i], "-output") || !strcmp(argv[i], "-o") ) {
	 strcpy (output_file, argv[++i]);
      }

      else if ( !strcmp(argv[i], "-R")) {
         rename_file = 0;
      }

      else if ( !strcmp(argv[i], "-version") || !strcmp(argv[i], "-v") ) {
	fprintf ( stderr,"Version number of gksm2ps: Mod %s / %s\n", mod,gksm2ps_version);
	exit (0);
      }

      else if ( !strcmp(argv[i], "-C")) {
         use_cmyk = 1;
      }

      else {
         if ( nfile == MAXFILE) {
            fprintf( stderr, " Maximum number of files, %d, exceeded.  Exiting.\n", MAXFILE );
            exit(1);
         }
	 strcpy(file[nfile++],argv[i]);
      }
   }
   
   if (nfile == 0) {
      fprintf( stderr, "Metafile name: ");
      scanf("%s",file[nfile++]);
      if (file[0][0] == '\0')
         exit(0);
   }


/* Get the current time */
   now = time(NULL);
   strftime (stamp,8,"_%H%M%S",localtime(&now));

/* Use correct line bundle */
   for (i=0; i<MAX_BUNDL_TBL; i++) 
      line_bundle[i] = (color_lines) ? color_line_bundle[i]:bw_line_bundle[i];

/* Open gks */
   gopengks (stderr, 0);

/* Open the metafile */            
   for (i=0; i<nfile; i++) {
      meta_id = 5;
      if (gks_stat=gopenws (meta_id,file[i],"MI") != 0) {
         fprintf ( stderr, "Error %d opening GKS metafile %s -- file not found\n", gks_stat, file[i]);
         exit(1);
      }

      /* Check version number of metafile against gksm2ps */
      if ((meta_file = fopen (file[i],"r")) == NULL) {
	fprintf ( stderr, "Can't open %s\n", meta_file);
	exit (1);
      }

      fgets (metafile_header, 45, meta_file);
      metafile_version = metafile_header + 24;
      if (strcmp(gksm2ps_version, metafile_version)) 
	fprintf (stderr, "Warning: Metafile version (%s) of '%s' differs\n     from gksm2ps version (%s).  Continuing ...\n", 
		 metafile_version, file[i], gksm2ps_version);
      fclose (meta_file);


      /* Using Xpreview option ? */
      if (xpreview_set)
      {
       	xpreview(meta_id, i, nfile, file[i]);
      }
      else
      {

	/* Set up border, offset info */
	scaling_completed = 0;
	pgwidth  = 612;
	pgheight = 792;     
	
	if (!user_set_geometry)
	  {
	    xoff = yoff = 0;
	    xborder = yborder = 18;
	    
	    /* phaser with transfer sheets cheats you on size */
	    if (device == phaser) {
	      xborder = 18 + 15;
	      yborder = 18 + 20;
	    }
	  } 
	else 
	  xborder = yborder = 0; 

	/* Initialize the PostScript output file*/
	if (i == 0) 
	  {
	    if ((ps_output = fopen (output_file,"w")) == NULL) 
	      {
		fprintf ( stderr, "Can't open %s\n", output_file);
		exit (1);
	      }

	    /* Get Bounding Box info by reading metafile items type 71 & 72 */
	    if (nfile == 1)
	      {
		find_BB( meta_id );
		gclosews ( meta_id );
		if (gks_stat=gopenws (meta_id,file[i],"MI") != 0) 
		  {
		    fprintf ( stderr, "Error %d opening GKS metafile %s -- file not found\n", gks_stat, file[i]);
		exit(1);
		  }
	      }
	    initpage();
	  }

	/* Omit initmatrix if only 1 file -- not needed and EPS non-compliant */
	if (nfile != 1) fprintf(ps_output, "initmatrix\n");

	ps_trans_meta( meta_id );
	fprintf(ps_output, "showpage\n");

	if (rename_file) time_stamp (file[i], stamp);
      }

      gclosews ( meta_id );
    }

/* Close the output device */
   fclose (ps_output);
   gclosegks();

}
