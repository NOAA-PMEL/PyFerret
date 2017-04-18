#ifndef GRDEL_H_
#define GRDEL_H_

#include <Python.h>

#ifdef VERBOSEDEBUG
#include <stdio.h>
extern FILE *debuglogfile;
#endif

/*
 * Code for a delegate "object" for graphics calls used by Ferret/PlotPlus.
 * Coordinates are (usually) from the bottom left corner increasing to the
 * top right corner.
 *
 * All the graphics functions in this library simply call the appropriate
 * function(s)/method(s) of the graphics library/object being used for that
 * window.
 *
 * Note that in double-precision Ferret, the PlotPlus code is still single
 * precision.  Thus all the C functions still use float, and the Fortran
 * functions use REAL*4, for their arguments.  However, the bindings 
 * functions use double precision; thus the C functions convert any float
 * arguments to double for calling the bindings function.
 */

/*
 * Opaque general graphics delegate object type.
 * Valid objects are not NULL.
 */
typedef void * grdelType;

/*
 * Alias of int type to clarify when the value should be considered
 * a boolean type.  Non-zero is considered "true", "yes", or "success".
 * Zero is considered "false", "no", or "failure".
 */
typedef int grdelBool;

/*
 * Declaration of a structure to be defined
 * (since cferbind.h includes this file).
 */
struct CFerBind_struct;

/*
 * Bindings for the window.  Only one of the elements
 * will be defined and the rest will be NULL.
 */
typedef struct BindObj_struct {
    struct CFerBind_struct *cferbind;
    PyObject *pyobject;
} BindObj;

/*
 * Global string for error messages.  When a function returns an error
 * value, some explanatory message should be assigned to this string.
 */
extern char grdelerrmsg[2048];

/*
 * Fortran interface for retrieving the error message.
 */
void fgderrmsg_(char *errmsg, int *errmsglen);

/*
 * Prototypes of some Fortran functions called by C functions.
 */
void fgd_bkgcolor_(int *windowid, int *colorindex);
void fgd_get_window_size_(float *width, float *height);
void fgd_get_view_limits_(float *lftfrc, float *rgtfrc,
                          float *btmfrc, float *topfrc,
                          float *lftcrd, float *rgtcrd,
                          float *btmcrd, float *topcrd);
void fgd_getdpi_(int *windowid, float *dpix, float *dpiy);
void fgd_gswkvp_(int *windowid, float *xmin, float *xmax,
                                float *ymin, float *ymax);
void fgd_set_unmapped_default_(int *pngonly);
void fgd_set_transparency_(int *transparent);
void fgd_getanimate_(int *inanimation);

/*
 * "Window" refers to the full canvas; however, no drawing, except possibly
 * clearing the window, is done on the full window.  Instead, a "View" of
 * the Window is "begun", and drawing is done in that View of the Window.
 */

grdelType grdelWindowCreate(const char *engine, int enginelen,
                            const char *title, int titlelen, 
                            grdelBool visible, grdelBool noalpha,
                            grdelBool rasteronly);
const BindObj *grdelWindowVerify(grdelType window);
grdelBool grdelWindowDelete(grdelType window);
grdelBool grdelWindowClear(grdelType window, grdelType fillcolor);
grdelBool grdelWindowRedraw(grdelType window, grdelType fillcolor);
grdelBool grdelWindowSetImageName(grdelType window, const char *imagename,
                     int imgnamelen, const char *formatname, int fmtnamelen);
grdelBool grdelWindowUpdate(grdelType window);
grdelBool grdelWindowSetAntialias(grdelType window, int antialias);
grdelBool grdelWindowSetSize(grdelType window, float width, float height);
grdelBool grdelWindowSetScale(grdelType window, float scale);
grdelBool grdelWindowSetVisible(grdelType window, grdelBool visible);
grdelBool grdelWindowSave(grdelType window, const char *filename,
                          int filenamelen, const char *fileformat,
                          int formatlen, grdelBool transparentbkg,
                          float xinches, float yinches, 
                          int xpixels, int ypixels,
                          void **annotations, int numannotations);
grdelBool grdelWindowScreenInfo(grdelType window, float *dpix, float *dpiy,
                                int *screenwidth, int* screenheight);
grdelBool grdelWindowSetWidthFactor(grdelType window, float widthfactor);

/*
 * Fortran interfaces for the Window functions.
 */
void fgdwincreate_(void **window, char *engine, int *enginelen,
                   char *title, int *titlelen, int *visible, 
                   int *noalpha, int *rasteronly);
void fgdwindelete_(int *success, void **window);
void fgdwinclear_(int *success, void **window, void **fillcolor);
void fgdwinredraw_(int *success, void **window, void **fillcolor);
void fgdwinupdate_(int *success, void **window);
void fgdwinsetantialias_(int *success, void **window, int *antialias);
void fgdwinsetsize_(int *success, void **window, float *width, float *height);
void fgdwinsetscale_(int *success, void **window, float *scale);
void fgdwinsetvis_(int *success, void **window, int *visible);
void fgdwinsave_(int *success, void **window, char *filename, int *namelen,
                 char *fileformat, int *formatlen, int *tranparentbkg,
                 float *xinches, float *yinches, int *xpixels, int *ypixels,
                 void **firststr, int *numstr);
void fgdwinscreeninfo_(int *success, void **window, float *dpix, float *dpiy,
                       int *screenwidth, int* screenheight);
void fgdwinsetwidthfactor_(int *success, void **window, float *widthfactor);

/*
 * A "View" refers to a rectangular subsection of the Window, possibly
 * the full Window.  Drawing is performed after defining a View; however,
 * coordinates are given in "device units" (pixels, using the current
 * Window DPI, from the top left corner).  The defined View is used to
 * set the clipping rectangle, when desired.  When drawing in this View 
 * is complete, the View is "ended".  Only one view can be active at a 
 * time, so switching between views requires ending one view and beginning
 * another view.
 */

grdelBool grdelWindowViewBegin(grdelType window,
                               float leftfrac, float bottomfrac,
                               float rightfrac, float topfrac,
                               int clipit);
grdelBool grdelWindowViewClip(grdelType window, int clipit);
grdelBool grdelWindowViewEnd(grdelType window);

/*
 * Fortran interfaces for the Window View functions.
 */
void fgdviewbegin_(int *success, void **window,
                   float *leftfrac, float *bottomfrac,
                   float *rightfrac, float *topfrac,
                   int *clipit);
void fgdviewclip_(int *success, void **window, int *clipit);
void fgdviewend_(int *success, void **window);

/*
 * A segment is an collection of drawing commands with an ID.  Drawing
 * commands in a segment can be deleted and the image recreated from
 * the remaining drawing commands.
 */
grdelBool grdelWindowSegmentBegin(grdelType window, int segid);
grdelBool grdelWindowSegmentEnd(grdelType window);
grdelBool grdelWindowSegmentDelete(grdelType window, int segid);

/*
 * Fortran interfaces for the Window Segment functions.
 */
void fgdsegbegin_(int *success, void **window, int *segid);
void fgdsegend_(int *success, void **window);
void fgdsegdelete_(int *success, void **window, int *segid);

void grdelGetTransformValues(double *my, double *sx, double *sy,
                                         double *dx, double *dy);

/*
 * All Color, Font, Pens, Brush, or Symbol objects can only be used
 * in the Window from which they were created.
 */

grdelType grdelColor(grdelType window, float redfrac,
               float greenfrac, float bluefrac, float opaquefrac);
grdelType grdelColorVerify(grdelType color, grdelType window);
grdelBool grdelColorDelete(grdelType color);

grdelType grdelFont(grdelType window, const char *familyname,
               int familynamelen, float fontsize, grdelBool italic,
               grdelBool bold, grdelBool underlined);
grdelType grdelFontVerify(grdelType font, grdelType window);
grdelBool grdelFontDelete(grdelType font);

grdelType grdelPen(grdelType window, grdelType color,
               float width, const char *style, int stylelen,
               const char *capstyle, int capstylelen,
               const char *joinstyle, int joinstylelen);
grdelType grdelPenVerify(grdelType pen, grdelType window);
grdelBool grdelPenReplaceColor(grdelType brush, grdelType color);
grdelBool grdelPenDelete(grdelType pen);

grdelType grdelBrush(grdelType window,  grdelType color,
               const char *style, int stylelen);
grdelType grdelBrushVerify(grdelType brush, grdelType window);
grdelBool grdelBrushReplaceColor(grdelType brush, grdelType color);
grdelBool grdelBrushDelete(grdelType brush);

grdelType grdelSymbol(grdelType window, const char *symbolname,
               int symbolnamelen);
grdelType grdelSymbolVerify(grdelType symbol, grdelType window);
grdelBool grdelSymbolDelete(grdelType symbol);

/*
 * Fortran interfaces for the Color, Font, Pen, Brush, and Symbol functions.
 */
void fgdcolor_(void **color, void **window, float *redfrac,
               float *greenfrac, float *bluefrac, float *opaquefrac);
void fgdcolordel_(int *success, void **color);

void fgdfont_(void **font, void **window, char *familyname, int *namelen,
               float *fontsize, int *italic, int *bold, int *underlined);
void fgdfontdel_(int *success, void **font);

void fgdpen_(void **pen, void **window, void **color, float *width,
               char *style, int *stylelen, char *capstyle, int *capstylelen,
               char *joinstyle, int *joinstylelen);
void fgdpenreplacecolor_(int *success, void **pen, void **color);
void fgdpendel_(int *success, void **pen);

void fgdbrush_(void **brush, void **window, void **color,
               char *style, int *stylelen);
void fgdbrushreplacecolor_(int *success, void **brush, void **color);
void fgdbrushdel_(int *success, void **brush);

void fgdsymbol_(void **symbol, void **window, char *symbolname, int *namelen);
void fgdsymboldel_(int *success, void **symbol);

/*
 * Drawing commands
 */

grdelBool grdelDrawMultiline(grdelType window, const float ptsx[],
               const float ptsy[], int numpts, grdelType pen);
grdelBool grdelDrawPoints(grdelType window, const float ptsx[],
               const float ptsy[], int numpts, grdelType symbol,
               grdelType color, float ptsize);
grdelBool grdelDrawPolygon(grdelType window, const float ptsx[],
               const float ptsy[], int numpts, grdelType brush,
               grdelType pen);
grdelBool grdelDrawRectangle(grdelType window, float left, float bottom,
               float right, float top, grdelType brush, grdelType pen);
grdelBool grdelDrawText(grdelType window, const char *text, int textlen,
               float startx, float starty, grdelType font, grdelType color,
               float rotate);

/*
 * Fortran interfaces for the drawing commands.
 */
void fgddrawmultiline_(int *success, void **window, float ptsx[],
               float ptsy[], int *numpts, void **pen);
void fgddrawpoints_(int *success, void **window, float ptsx[],
               float ptsy[], int *numpts, void **symbol,
               void **color, float *ptsize);
void fgddrawpolygon_(int *success, void **window, float ptsx[],
               float ptsy[], int *numpts, void **brush, void **pen);
void fgddrawrect_(int *success, void **window, float *left, float *bottom,
               float *right, float *top, void **brush, void **pen);
void fgddrawtext_(int *success, void **window, char *text, int *textlen,
               float *startx, float *starty, void **font, void **color,
               float *rotate);

#endif /* GRDEL_H_ */

