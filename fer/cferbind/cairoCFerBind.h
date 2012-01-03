#ifndef CAIRO_CFERBIND_H_
#define CAIRO_CFERBIND_H_

/* Make sure Python.h is always included first */
#include <Python.h>
#include <cairo/cairo.h>
/* Use of grdelBool (int) and grdelType (void *) is just to clarify intent */
#include "grdel.h"

/* Size of filename string arrays */
#define CCFB_NAME_SIZE 512

/* DPI to use for raster images (PNG) */
#define CCFB_RASTER_DPI 300

/* DPI to use for vector images (not PNG) */
#define CCFB_VECTOR_DPI 1200

/* Factor for converting pixels to points */
#define CCFB_POINTS_PER_PIXEL (72.0 / (double) CCFB_VECTOR_DPI);

typedef enum CCFBImageFormat_enum {
    CCFBIF_PNG = 0,
    CCFBIF_PDF,
    CCFBIF_EPS,
    CCFBIF_SVG,
} CCFBImageFormat;

typedef struct CCFBSides_struct {
    double left;
    double right;
    double top;
    double bottom;
} CCFBSides;

extern const char *CCFBColorId;
typedef struct CCFBColor_Struct {
    const char *id;
    double redfrac;
    double greenfrac;
    double bluefrac;
    double opaquefrac;
} CCFBColor;

extern const char *CCFBPenId;
typedef struct CCFBPen_Struct {
    const char *id;
    CCFBColor color;
    double width;
    int    numdashes;
    double dashes[8];
    cairo_line_cap_t captype;
    cairo_line_join_t jointype;
} CCFBPen;

typedef struct CairoCFerBindData_struct {
    /* image size in pixels */
    int imagewidth;
    int imageheight;
    int minsize;
    /* clearing color */
    CCFBColor lastclearcolor;
    /* image filename and format */
    char imagename[CCFB_NAME_SIZE];
    CCFBImageFormat imageformat;
    /* Anti-aliasing value for non-text elements */
    int antialias;
    /*
     * data for recreating the current view
     * only used to define the clipping rectangle
     */
    CCFBSides fracsides;
    int clipit;
    /*
     * The surface and context are not created until a view is created,
     * and thus drawing is about to begin.  Ferret will modify the above
     * values, possibly multiple times, prior to the start of drawing.
     */
    cairo_surface_t *surface;
    cairo_t *context;
    /* Flag that something has been drawn to the current surface */
    int somethingdrawn;
} CairoCFerBindData;

grdelBool cairoCFerBind_setImageName(CFerBind *self, char *imagename,
                        int imgnamelen, char *formatname, int fmtnamelen);
grdelBool cairoCFerBind_deleteWindow(CFerBind *self);
grdelBool cairoCFerBind_setAntialias(CFerBind *self, int antialias);
grdelBool cairoCFerBind_beginView(CFerBind *self, double lftfrac, double btmfrac,
                                  double rgtfrac, double topfrac, int clipit);
grdelBool cairoCFerBind_clipView(CFerBind *self, int clipit);
grdelBool cairoCFerBind_endView(CFerBind *self);
grdelBool cairoCFerBind_updateWindow(CFerBind *self);
grdelBool cairoCFerBind_clearWindow(CFerBind *self, grdelType fillcolor);
double *  cairoCFerBind_windowDpi(CFerBind *self);
grdelBool cairoCFerBind_resizeWindow(CFerBind *self, double width, double height);
grdelBool cairoCFerBind_showWindow(CFerBind *self, int visible);
grdelBool cairoCFerBind_saveWindow(CFerBind *self, char *filename, int namelen,
                                   char *formatname, int fmtnamelen, int transbkg);
grdelType cairoCFerBind_createColor(CFerBind *self, double redfrac,
                        double greenfrac, double bluefrac, double opaquefrac);
grdelBool cairoCFerBind_deleteColor(CFerBind *self, grdelType color);
grdelType cairoCFerBind_createFont(CFerBind *self, char *familyname, int namelen,
                        double fontsize, int italic, int bold, int underlined);
grdelBool cairoCFerBind_deleteFont(CFerBind *self, grdelType font);
grdelType cairoCFerBind_createPen(CFerBind *self, grdelType color, double width,
                                  char *style, int stlen, char *capstyle,
                                  int capstlen, char *joinstyle, int joinstlen);
grdelBool cairoCFerBind_deletePen(CFerBind *self, grdelType pen);
grdelType cairoCFerBind_createBrush(CFerBind *self, grdelType color,
                                    char *style, int stlen);
grdelBool cairoCFerBind_deleteBrush(CFerBind *self, grdelType brush);
grdelType cairoCFerBind_createSymbol(CFerBind *self, char *symbolname, int namelen);
grdelBool cairoCFerBind_deleteSymbol(CFerBind *self, grdelType symbol);
grdelBool cairoCFerBind_drawMultiline(CFerBind *self, double ptsx[], double ptsy[],
                                      grdelType pen);
grdelBool cairoCFerBind_drawPoints(CFerBind *self, double ptsx[], double ptsy[],
                                   grdelType symbol, grdelType color, double ptsize);
grdelBool cairoCFerBind_drawPolygon(CFerBind *self, double ptsx[], double ptsy[],
                                    grdelType brush, grdelType pen);
grdelBool cairoCFerBind_drawRectangle(CFerBind *self, double left, double bottom,
                        double right, double top, grdelType brush, grdelType pen);
grdelBool cairoCFerBind_drawMulticoloredRectangle(CFerBind *self, double left,
                        double bottom, double right, double top,
                        int numrows, int numcols, grdelType colors[]);
grdelBool cairoCFerBind_drawText(CFerBind *self, char *text, int textlen,
                                 double startx, double starty, grdelType font,
                                 grdelType color, double rotation);

#endif
