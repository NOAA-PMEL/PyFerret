/*
 *		Copyright IBM Corporation 1989
 *
 *                      All Rights Reserved
 *
 * Permission to use, copy, modify, and distribute this software and its
 * documentation for any purpose and without fee is hereby granted,
 * provided that the above copyright notice appear in all copies and that
 * both that copyright notice and this permission notice appear in
 * supporting documentation, and that the name of IBM not be
 * used in advertising or publicity pertaining to distribution of the
 * software without specific, written prior permission.
 *
 * IBM DISCLAIMS ALL WARRANTIES WITH REGARD TO THIS SOFTWARE, INCLUDING
 * ALL IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS, IN NO EVENT SHALL
 * IBM BE LIABLE FOR ANY SPECIAL, INDIRECT OR CONSEQUENTIAL DAMAGES OR
 * ANY DAMAGES WHATSOEVER RESULTING FROM LOSS OF USE, DATA OR PROFITS,
 * WHETHER IN AN ACTION OF CONTRACT, NEGLIGENCE OR OTHER TORTIOUS ACTION,
 * ARISING OUT OF OR IN CONNECTION WITH THE USE OR PERFORMANCE OF THIS
 * SOFTWARE.
 *
 * FORTRAN to C binding for XGKS
 *
 * Todd Gill
 * Dave Owens
 * TCS Development
 * Cambridge MA
 *
 * $Id$
 * $__Header$
 */

#ifndef FORTMAC_H_INCLUDED
#define FORTMAC_H_INCLUDED

#define ASPECTSOURCE(x,y) \
          if (((x)<(int)FORT_GBUNDL)||((x)>(int)FORT_GINDIV)) \
             {(void)gerrorhand(2000,y,(errfp)); return;}
#define CLEARCONTROLFLAG(x,y) \
          if (((int)(x)<(int)FORT_GCONDI)||((int)(x)>(int)FORT_GALWAY)) \
             {(void)gerrorhand(2000,y,(errfp)); return;}
#define CLIPPINGINDICATOR(x,y) \
          if (((x)<(int)FORT_GNCLIP)||((x)>(int)FORT_GCLIP)) \
             {(void)gerrorhand(2000,y,(errfp)); return;}
#define COLOURAVAILABLE(x,y) \
          if (((x)<(int)FORT_GMONOC)||((x)>(int)FORT_GCOLOR)) \
             {(void)gerrorhand(2000,y,(errfp)); return;}
#define COORDINATESWITCH(x,y) \
          if (((x)<(int)FORT_GWC)||((x)>(int)FORT_GNDC)) \
             {(void)gerrorhand(2000,y,(errfp)); return;}
#define DEFERRALMODE(x,y) \
          if (((int)(x)<(int)FORT_GASAP)||((int)(x)>(int)FORT_GASTI)) \
             {(void)gerrorhand(2000,y,(errfp)); return;}
#define DETECTABILITY(x,y) \
          if (((x)<(int)FORT_GUNDET)||((x)>(int)FORT_GDETEC)) \
             {(void)gerrorhand(2000,y,(errfp)); return;}
#define DEVICECOORDINATEUNITS(x,y) \
          if (((x)<(int)FORT_GMETRE)||((x)>(int)FORT_GOTHU))  \
             {(void)gerrorhand(2000,y,(errfp)); return;}
#define DISPLAYSURFACEEMPTY(x,y) \
          if (((x)<(int)FORT_GNEMPT)||((x)>(int)FORT_GEMPTY)) \
             {(void)gerrorhand(2000,y,(errfp)); return;}
#define DYNAMICMODIFICATION(x,y) \
          if (((x)<(int)FORT_GIRG)||((x)>(int)FORT_GIMM)) \
             {(void)gerrorhand(2000,y,(errfp)); return;}
#define ECHOSWITCH(x,y) \
          if (((x)<(int)FORT_GNECHO)||((x)>(int)FORT_GECHO)) \
             {(void)gerrorhand(2000,y,(errfp)); return;}
#define FILLAREAINTERIORSTYLE(x,y) \
          if (((x)<(int)FORT_GHOLLO)||((x)>(int)FORT_GHATCH)) \
             {(void)gerrorhand(2000,y,(errfp)); return;}
#define HIGHLIGHTING(x,y) \
          if (((x)<(int)FORT_GNORML)||((x)>(int)FORT_GHILIT)) \
             {(void)gerrorhand(2000,y,(errfp)); return;}
#define INPUTDEVICESTATUS(x,y) \
          if (((x)<(int)FORT_GNONE)||((x)>(int)FORT_GNCHOI)) \
             {(void)gerrorhand(2000,y,(errfp)); return;}
#define INPUTCLASS(x,y) \
          if (((x)<(int)FORT_GNCLAS)||((x)>(int)FORT_GSTRIN)) \
             {(void)gerrorhand(2000,y,(errfp)); return;}
#define IMPLICITREGENERATIONMODE(x,y) \
          if (((int)(x)<(int)FORT_GSUPPD)||((int)(x)>(int)FORT_GALLOW)) \
             {(void)gerrorhand(2000,y,(errfp)); return;}
#define LEVELOFGKS(x,y) \
          if (((x)<((int)FORT_GLMA))||((x)>(int)FORT_GL2C)) \
             {(void)gerrorhand(2000,y,(errfp)); return;}
#define NEWFRAMEACTIONNECESSARY(x,y) \
          if (((x)<(int)FORT_GNO)||((x)>(int)FORT_GYES)) \
             {(void)gerrorhand(2000,y,(errfp)); return;}
#define OPERATINGMODE(x,y) \
          if (((x)<(int)FORT_GREQU)||((x)>(int)FORT_GEVENT)) \
             {(void)gerrorhand(2000,y,(errfp)); return;}
#define OPERATINGSTATEVALUE(x,y) \
          if (((x)<(int)FORT_GGKCL)||((x)>(int)FORT_GSGOP)) \
             {(void)gerrorhand(2000,y,(errfp)); return;}
#define PRESENCEOFINVALIDVALUES(x,y) \
          if (((x)<(int)FORT_GABSNT)||((x)>(int)FORT_GPRSNT)) \
             {(void)gerrorhand(2000,y,(errfp)); return;}
#define REGENERATIONFLAG(x,y) \
          if (((int)(x)<(int)FORT_GPOSTP)||((int)(x)>(int)FORT_GPERFO)) \
             {(void)gerrorhand(2000,y,(errfp)); return;}
#define RELATIVEINPUTPRIORITY(x,y) \
          if (((x)<(int)FORT_GHIGHR)||((x)>(int)FORT_GLOWER)) \
             {(void)gerrorhand(2000,y,(errfp)); return;}
#define SIMULTANEOUSEVENTSFLAG(x,y) \
          if (((x)<(int)FORT_GNMORE)||((x)>(int)FORT_GMORE)) \
             {(void)gerrorhand(2000,y,(errfp)); return;}
#define STOREAGEFLAG(x,y)\
          if (((x)<(int)FORT_GNO) || ((x)>(int)FORT_GYES)) \
             {(void)gerrorhand(2000,y,(errfp)); return;}
#define TEXTALIGNMENTHORIZONTAL(x,y) \
          if (((x)<(int)FORT_GAHNOR)||((x)>(int)FORT_GARITE)) \
             {(void)gerrorhand(2000,y,(errfp)); return;}
#define TEXTALIGNMENTVERTICAL(x,y) \
          if (((x)<(int)FORT_GAVNOR)||((x)>(int)FORT_GABOTT)) \
             {(void)gerrorhand(2000,y,(errfp)); return;}
#define TEXTPATH(x,y) \
          if (((x)<(int)FORT_GRIGHT)||((x)>(int)FORT_GDOWN)) \
             {(void)gerrorhand(2000,y,(errfp)); return;}
#define TEXTPRECISION(x,y) \
          if (((x)<(int)FORT_GSTRP)||((x)>(int)FORT_GSTRKP)) \
             {(void)gerrorhand(2000,y,(errfp)); return;}
#define TYPEOFRETURNEDVALUES(errind,x,y) \
          if (((x)<(int)FORT_GSET)||((x)>(int)FORT_GREALI)) { \
	      if ((errind)) \
		  *(errind) = (int) (2000);(void)gerrorhand(2000,y,(errfp)); \
		  return; \
	  }
#define UPDATESTATENOTPENDING(x,y) \
          if (((x)<(int)FORT_GNPEND)||((x)>(int)FORT_GPEND)) \
             {(void)gerrorhand(2000,y,(errfp)); return;}
#define VECTORRASTEROTHERTYPE(x,y) \
          if (((x)<(int)FORT_GVECTR)||((x)>(int)FORT_GOTHWK)) \
             {(void)gerrorhand(2000,y,(errfp)); return;}
#define VISIBILITY(x,y) \
          if (((x)<(int)FORT_GINVIS)||((x)>(int)FORT_GVISI)) \
             {(void)gerrorhand(2000,y,(errfp)); return;}
#define WORKSTATIONCATEGORY(x,y) \
          if (((x)<(int)FORT_GOUTPT)||((x)>(int)FORT_GMI)) \
             {(void)gerrorhand(2000,y,(errfp)); return;}
#define WORKSTATIONSTATE(x,y) \
          if (((x)<(int)FORT_GINACT)||((x)>(int)FORT_GACTIV)) \
             {(void)gerrorhand(2000,y,(errfp)); return;}
#define LISTOFGDPATTRIBUTES(x,y) \
          if (((x)<(int)FORT_GPLATT)||((x)>(int)FORT_GFAATT)) \
             {(void)gerrorhand(2000,y,(errfp)); return;}
#define LINETYPE(x,y) \
          if (((x)<(int)FORT_GLSOLI)||((x)>(int)FORT_GLDASD)) \
             {(void)gerrorhand(2000,y,(errfp)); return;}
#define MARKERTYPE(x,y) \
          if (((x)<(int)FORT_GPOINT)||((x)>(int)FORT_GXMARK)) \
             {(void)gerrorhand(2000,y,(errfp)); return;}
#define ATTRIBUTECONTROLFLAG(x,y) \
          if (((x)<(int)FORT_GCURNT)||((x)>(int)FORT_GSPEC)) \
             {(void)gerrorhand(2000,y,(errfp)); return;}
#define POLYLINEFILLAREACONTROLFLAG(x,y) \
          if (((x)<(int)FORT_GPLINE)||((x)>(int)FORT_GFILLA)) \
             {(void)gerrorhand(2000,y,(errfp)); return;}
#define INITIALCHOICEPROMPTFLAG(x,y) \
          if (((x)<(int)FORT_GPROFF)||((x)>(int)FORT_GPRON)) \
             {(void)gerrorhand(2000,y,(errfp)); return;}
#define VALIDTYPE(errind,x,min,max,fctid) \
          if (((x)<min)||((x)>max)) \
             {if ((errind) != NULL) *(errind) = (int) (22); \
	     (void)gerrorhand(22,fctid,(errfp)); return;}
#define VALIDMEMBER(errind,x,min,max,fctid) \
          if (((x)<min)||((x)>max)) \
             {if (errind) *(errind) = (int) (2002); \
	     (void)gerrorhand(2002,fctid,(errfp)); return;}
#define CHECKMAGICNUM(datarec,fctid)           \
          if ( ((Gdatarec *) datarec)->magicnum != MAGICNUMBER) \
             {(void)gerrorhand(2003,fctid,(errfp));return;}

#undef  MIN
#define MIN(x,y)	((x) < (y) ? (x) : (y))

/* move values from a 2X3 array into a 1X6 array */
#define MOVE_ARRAY_2X3_TO_1X6(from,to) \
   (to)[0] = (float) (from)[0][0]; \
   (to)[1] = (float) (from)[1][0]; \
   (to)[2] = (float) (from)[0][1]; \
   (to)[3] = (float) (from)[1][1]; \
   (to)[4] = (float) (from)[0][2]; \
   (to)[5] = (float) (from)[1][2];

/* move values from a 1X6 array into a 2X3 array */
#define MOVE_ARRAY_1X6_TO_2X3(from,to) \
   (to)[0][0] = (Gfloat) (from)[0]; \
   (to)[1][0] = (Gfloat) (from)[1]; \
   (to)[0][1] = (Gfloat) (from)[2]; \
   (to)[1][1] = (Gfloat) (from)[3]; \
   (to)[0][2] = (Gfloat) (from)[4]; \
   (to)[1][2] = (Gfloat) (from)[5];

/* convert column major matrix to row major matrix */
/* NOTE: dimx & dimy are the dimentions of "to"    */
#define CHANGE_ROW_TO_COL_MAJOR_I(from,to,dimx,dimy) \
  {                                                  \
  int col,row;                                       \
  for (col = 0; col < (dimy); col++)                 \
    for (row = 0; row < (dimx); row++)               \
      *((to) + (col*(dimx)+row)) = (int) *((from) + (row*(dimy)+col)); \
  }

/* convert column major matrix to row major matrix */
/* NOTE: dimx & dimy are the dimentions of "to"    */
#define CHANGE_ROW_TO_COL_MAJOR_F(from,to,dimx,dimy) \
  {                                                  \
  int col,row;                                       \
  for (col = 0; col < (dimy); col++)                 \
    for (row = 0; row < (dimx); row++)               \
      *((to) + (col*(dimx)+row)) = (float) *((from) + (row*(dimy)+col)); \
  }

/* convert column major matrix to row major matrix             */
/* NOTE: rows & cols are the FORTRAN dimentions: TO(ROWS,COLS) */
#define CHANGE_COL_TO_ROW_MAJOR_I(from,to,rows,cols) \
  {                                                  \
  int c,r;                                           \
  for (c = 0; c < (cols); c++)                       \
    for (r = 0; r < (rows); r++)                     \
      *((to) + (c*(rows)+r)) = (Gint) *((Gint *)(from) + (r*(cols)+c)); \
  }

/* convert column major matrix to row major matrix             */
/* NOTE: rows & cols are the FORTRAN dimentions: TO(ROWS,COLS) */
#define CHANGE_COL_TO_ROW_MAJOR_F(from,to,rows,cols) \
  {                                                  \
  int c,r;                                           \
  for (c = 0; c < (cols); c++)                       \
    for (r = 0; r < (rows); r++)                     \
      *((to) + (c*(rows)+r)) = (Gfloat) *((Gfloat *)(from) + (r*(cols)+c)); \
  }

/*
 * Define a "realloc" macro that checks whether or not the pointer argument
 * is NULL and does the right thing.  NB: This macro is DANGEROUS because it
 * evaluates the pointer twice; hence, we capitalize its name.
 */
#define REALLOC(p,n)	((p) == NULL ? malloc(n) : realloc((p),(n)))

#endif	/* FORTMAC_H_INCLUDED not defined */
