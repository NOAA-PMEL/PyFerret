#ifndef CAIRO_CFERBIND_H_
#define CAIRO_CFERBIND_H_

/* Make sure Python.h is always included first */
#include <Python.h>
#include <cairo/cairo.h>
#ifdef USEPANGOCAIRO
#include <pango/pangocairo.h>
#endif
/* Use of grdelBool (int) and grdelType (void *) is just to clarify intent */
#include "grdel.h"

/* Size of filename string arrays */
#define CCFB_NAME_SIZE 512

typedef enum CCFBImageFormat_enum {
    CCFBIF_PNG = 0,
    CCFBIF_PDF,
    CCFBIF_PS,
    CCFBIF_SVG,
    CCFBIF_REC,
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

extern const char *CCFBBrushId;
typedef struct CCFBBrush_Struct {
    const char *id;
    CCFBColor color;
    cairo_pattern_t *pattern;
} CCFBBrush;

extern const char *CCFBSymbolId;
typedef struct CCFBSymbol_Struct {
    const char *id;
    cairo_path_t *path;
    grdelBool filled;
    char name[256];
} CCFBSymbol;

extern const char *CCFBFontId;
typedef struct CCFBFont_Struct {
    const char *id;
#ifdef USEPANGOCAIRO
    PangoFontDescription *fontdesc;
#else
    cairo_font_face_t *fontface;
    double fontsize;
#endif
    int underline;
} CCFBFont;

/* Structure for creating a linked list of image or recording surfaces */
typedef struct CCFBPicture_Struct {
    struct CCFBPicture_Struct *next;
    cairo_surface_t *surface;
    int segid;
} CCFBPicture;

typedef struct CairoCFerBindData_struct {
    double pixelsperinch;
    /* image size in pixels */
    int imagewidth;
    int imageheight;
    int minsize;
    /* Scaling factor for line widths and symbol sizes */
    double widthfactor;
    /* clearing color */
    CCFBColor lastclearcolor;
    /* image filename and format */
    char imagename[CCFB_NAME_SIZE];
    CCFBImageFormat imageformat;
    /* Anti-alias non-text elements? */
    int antialias;
    /*
     * Never use colors with an alpha channel (ARGB32) ?
     * If false (zero), it will depend on the output format.
     */
    int noalpha;
    /* data for recreating the current view */
    CCFBSides fracsides;
    int clipit;
    /* Linked list of image or recording surfaces, with segment IDs */
    CCFBPicture *firstpic;
    CCFBPicture *lastpic;
    int segid;
    /*
     * The current surface and context.  These are not created until
     * a view is created and drawing is about to begin.  Ferret will
     * modify the above values, possibly multiple times, prior to the
     * start of drawing.
     */
    cairo_surface_t *surface;
    cairo_t *context;
    /* Flag that something has been drawn to the current surface */
    int somethingdrawn;
    /*
     * Flag that something about the image has changed since the last
     * update.  Only really used by the PipedImager engine.
     */
    int imagechanged;
    /*
     * The image displayer.
     * Only assigned and used by the PipedImager engine.
     */
    grdelType viewer;
    /*
     * Parameters for setting watermark image source file and display properties.
     */
    float xloc;
    float yloc;
    float scalefrac;
    float opacity;
    char  wmark_filename[CCFB_NAME_SIZE];
} CairoCFerBindData;

grdelBool cairoCFerBind_setImageName(CFerBind *self, const char *imagename,
                        int imgnamelen, const char *formatname, int fmtnamelen);
grdelBool cairoCFerBind_createSurface(CFerBind *self);
grdelBool cairoCFerBind_deleteWindow(CFerBind *self);
grdelBool cairoCFerBind_setAntialias(CFerBind *self, int antialias);
grdelBool cairoCFerBind_beginView(CFerBind *self, double lftfrac, double btmfrac,
                                  double rgtfrac, double topfrac, int clipit);
grdelBool cairoCFerBind_clipView(CFerBind *self, int clipit);
grdelBool cairoCFerBind_endView(CFerBind *self);
grdelBool cairoCFerBind_beginSegment(CFerBind *self, int segid);
grdelBool cairoCFerBind_endSegment(CFerBind *self);
grdelBool cairoCFerBind_deleteSegment(CFerBind *self, int segid);
grdelBool cairoCFerBind_updateWindow(CFerBind *self);
grdelBool cairoCFerBind_clearWindow(CFerBind *self, grdelType fillcolor);
grdelBool cairoCFerBind_redrawWindow(CFerBind *self, grdelType fillcolor);
grdelBool cairoCFerBind_windowScreenInfo(CFerBind *self, float *dpix, float *dpiy,
                                         int *screenwidth, int *screenheight);
grdelBool cairoCFerBind_setWindowDpi(CFerBind *self, double newdpi);
grdelBool cairoCFerBind_resizeWindow(CFerBind *self, double width, double height);
grdelBool cairoCFerBind_scaleWindow(CFerBind *self, double scale);
grdelBool cairoCFerBind_showWindow(CFerBind *self, int visible);
grdelBool cairoCFerBind_saveWindow(CFerBind *self, const char *filename, int namelen,
                                   const char *formatname, int fmtnamelen, int transbkg,
                                   double xinches, double yinches,
                                   int xpixels, int ypixels,
                                   void **annotations, int numannotations);
grdelType cairoCFerBind_createColor(CFerBind *self, double redfrac,
                        double greenfrac, double bluefrac, double opaquefrac);
grdelBool cairoCFerBind_deleteColor(CFerBind *self, grdelType color);
grdelType cairoCFerBind_createFont(CFerBind *self, const char *familyname, int namelen,
                        double fontsize, int italic, int bold, int underlined);
grdelBool cairoCFerBind_deleteFont(CFerBind *self, grdelType font);
grdelType cairoCFerBind_createPen(CFerBind *self, grdelType color, double width,
                                  const char *style, int stlen, const char *capstyle,
                                  int capstlen, const char *joinstyle, int joinstlen);
grdelBool cairoCFerBind_replacePenColor(CFerBind *self,
                                        grdelType pen, grdelType color);
grdelBool cairoCFerBind_deletePen(CFerBind *self, grdelType pen);
grdelType cairoCFerBind_createBrush(CFerBind *self, grdelType color,
                                    const char *style, int stlen);
grdelBool cairoCFerBind_replaceBrushColor(CFerBind *self,
                                          grdelType brush, grdelType color);
grdelBool cairoCFerBind_deleteBrush(CFerBind *self, grdelType brush);
grdelType cairoCFerBind_createSymbol(CFerBind *self, const char *symbolname, int namelen,
                        const float ptsx[], const float ptsy[], int numpts, grdelBool fill);
grdelBool cairoCFerBind_deleteSymbol(CFerBind *self, grdelType symbol);
grdelBool cairoCFerBind_setWidthFactor(CFerBind *self, double widthfactor);
grdelBool cairoCFerBind_drawMultiline(CFerBind *self, double ptsx[], double ptsy[],
                                      int numpts, grdelType pen);
grdelBool cairoCFerBind_drawPoints(CFerBind *self, double ptsx[], double ptsy[],
                                   int numpts, grdelType symbol, grdelType color,
                                   double symsize, grdelType highlight);
grdelBool cairoCFerBind_drawPolygon(CFerBind *self, double ptsx[], double ptsy[],
                                    int numpts, grdelType brush, grdelType pen);
grdelBool cairoCFerBind_drawRectangle(CFerBind *self, double left, double bottom,
                        double right, double top, grdelType brush, grdelType pen);
grdelBool cairoCFerBind_textSize(CFerBind *self, const char *text, int textlen,
                                 grdelType font, double *widthptr, double *heightptr);
grdelBool cairoCFerBind_drawText(CFerBind *self, const char *text, int textlen,
                                 double startx, double starty, grdelType font,
                                 grdelType color, double rotation);
grdelBool cairoCFerBind_setWaterMark(CFerBind *self, const char filename[], int len_filename,
                                     float xloc, float yloc, float scalefrac, float opacity);

#endif
