#ifndef CAIRO_CFERBIND_H_
#define CAIRO_CFERBIND_H_

#include <cairo/cairo.h>

typedef struct CairoCFerBindData_struct {
    cairo_surface_t *surface;
} CairoCFerBindData;

int      cairoCFerBind_deleteWindow(CFerBind *self);
int      cairoCFerBind_setAntialias(CFerBind *self, int antialias);
int      cairoCFerBind_beginView(CFerBind *self, double leftfrac, double bottomfrac,
                                 double rightfrac, double topfrac, int clipit);
int      cairoCFerBind_clipView(CFerBind *self, int clipit);
int      cairoCFerBind_endView(CFerBind *self);
int      cairoCFerBind_updateWindow(CFerBind *self);
int      cairoCFerBind_clearWindow(CFerBind *self, void *fillcolor);
double*  cairoCFerBind_windowDpi(CFerBind *self);
int      cairoCFerBind_resizeWindow(CFerBind *self, double width, double height);
int      cairoCFerBind_showWindow(CFerBind *self, int visible);
int      cairoCFerBind_saveWindow(CFerBind *self, char *filename, int namelen,
                                  char *format, int fmtlen, int transbkg);
void*    cairoCFerBind_createColor(CFerBind *self, double redfrac,
                                   double greenfrac, double bluefrac);
int      cairoCFerBind_deleteColor(CFerBind *self, void *color);
void*    cairoCFerBind_createFont(CFerBind *self, char *familyname, int namelen,
                             double fontsize, int italic, int bold, int underlined);
int      cairoCFerBind_deleteFont(CFerBind *self, void *font);
void*    cairoCFerBind_createPen(CFerBind *self, void *color, double width,
                                 char *style, int stlen, char *capstyle,
                                 int capstlen, char *joinstyle, int joinstlen);
int      cairoCFerBind_deletePen(CFerBind *self, void *pen);
void*    cairoCFerBind_createBrush(CFerBind *self, void *color,
                                   char *style, int stlen);
int      cairoCFerBind_deleteBrush(CFerBind *self, void *brush);
void*    cairoCFerBind_createSymbol(CFerBind *self, char *symbolname, int namelen);
int      cairoCFerBind_deleteSymbol(CFerBind *self, void *symbol);
int      cairoCFerBind_drawMultiline(CFerBind *self, double ptsx[], double ptsy[],
                                     void *pen);
int      cairoCFerBind_drawPoints(CFerBind *self, double ptsx[], double ptsy[],
                                  void *symbol, void *color, double ptsize);
int      cairoCFerBind_drawPolygon(CFerBind *self, double ptsx[], double ptsy[],
                                   void *brush, void *pen);
int      cairoCFerBind_drawRectangle(CFerBind *self, double left, double bottom,
                           double right, double top, void *brush, void *pen);
int      cairoCFerBind_drawMulticoloredRectangle(CFerBind *self, double left,
                                       double bottom, double right, double top, 
                                       int numrows, int numcols, void *colors[]);
int      cairoCFerBind_drawText(CFerBind *self, char *text, int textlen,
                                double startx, double starty, void *font,
                                void *color, double rotation);

#endif
