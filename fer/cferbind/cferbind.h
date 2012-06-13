#ifndef CFERBIND_H_
#define CFERBIND_H_

/* Make sure Python.h is always included first */
#include <Python.h>
/* Use of grdelBool (int) and grdelType (void *) is just to clarify intent */
#include "grdel.h"

/* Names of recognized engines */
extern const char *CairoCFerBindName;
extern const char *PyQtCairoCFerBindName;

/*
 * The bindings structure.  All functions should be defined in a bindings
 * instance, but a definition may just be a function that does nothing
 * if the action is not applicable or appropriate.
 *
 * The cferbind_createWindow function calls the appropriate createWindow
 * function to create the bindings instance and any other appropriate
 * initialization for this "Window".  Thus, there is no createWindow
 * function in these bindings since it should never need to be called.
 */
typedef struct CFerBind_struct {
     const char *enginename;
     void       *instancedata;
     grdelBool (*setImageName)(struct CFerBind_struct *self, const char *imagename,
                               int imgnamelen, const char *formatname, int fmtnamelen);
     grdelBool (*deleteWindow)(struct CFerBind_struct *self);
     grdelBool (*setAntialias)(struct CFerBind_struct *self, int antialias);
     grdelBool (*beginView)(struct CFerBind_struct *self,
                            double leftfrac, double bottomfrac,
                            double rightfrac, double topfrac, int clipit);
     grdelBool (*clipView)(struct CFerBind_struct *self, int clipit);
     grdelBool (*endView)(struct CFerBind_struct *self);
     grdelBool (*updateWindow)(struct CFerBind_struct *self);
     grdelBool (*clearWindow)(struct CFerBind_struct *self, grdelType fillcolor);
     double *  (*windowDpi)(struct CFerBind_struct *self);
     grdelBool (*resizeWindow)(struct CFerBind_struct *self,
                               double width, double height);
     grdelBool (*showWindow)(struct CFerBind_struct *self, int visible);
     grdelBool (*saveWindow)(struct CFerBind_struct *self, const char *filename,
                             int namelen, const char *formatname, int fmtnamelen,
                             int transbkg);
     grdelType (*createColor)(struct CFerBind_struct *self, double redfrac,
                              double greenfrac, double bluefrac, double opaquefrac);
     grdelBool (*deleteColor)(struct CFerBind_struct *self, grdelType color);
     grdelType (*createFont)(struct CFerBind_struct *self,
                             const char *familyname, int namelen, double fontsize,
                             int italic, int bold, int underlined);
     grdelBool (*deleteFont)(struct CFerBind_struct *self, grdelType font);
     grdelType (*createPen)(struct CFerBind_struct *self, grdelType color,
                            double width, const char *style, int stlen,
                            const char *capstyle, int capstlen,
                            const char *joinstyle, int joinstlen);
     grdelBool (*replacePenColor)(struct CFerBind_struct *self,
                                  grdelType pen, grdelType color);
     grdelBool (*deletePen)(struct CFerBind_struct *self, grdelType pen);
     grdelType (*createBrush)(struct CFerBind_struct *self,
                              grdelType color, const char *style, int stlen);
     grdelBool (*replaceBrushColor)(struct CFerBind_struct *self,
                                    grdelType brush, grdelType color);
     grdelBool (*deleteBrush)(struct CFerBind_struct *self, grdelType brush);
     grdelType (*createSymbol)(struct CFerBind_struct *self,
                               const char *symbolname, int namelen);
     grdelBool (*deleteSymbol)(struct CFerBind_struct *self, grdelType symbol);
     grdelBool (*drawMultiline)(struct CFerBind_struct *self, double ptsx[],
                                double ptsy[], int numpts, grdelType pen);
     grdelBool (*drawPoints)(struct CFerBind_struct *self, double ptsx[],
                             double ptsy[], int numpts, grdelType symbol,
                             grdelType color, double symsize);
     grdelBool (*drawPolygon)(struct CFerBind_struct *self, double ptsx[],
                              double ptsy[], int numpts, grdelType brush,
                              grdelType pen);
     grdelBool (*drawRectangle)(struct CFerBind_struct *self,
                                double left, double bottom, double right,
                                double top, grdelType brush, grdelType pen);
     grdelBool (*drawText)(struct CFerBind_struct *self, const char *text, int textlen,
                           double startx, double starty, grdelType font, grdelType color,
                           double rotation);
} CFerBind;

/*
 * Calls the appropriate createWindow function to create the bindings
 * instance and any other appropriate initialization for this "Window". 
 */
CFerBind *cferbind_createWindow(const char *enginename, int engnamelen,
                                const char *windowname, int winnamelen, int visible);

/* The createWindow function for the Cairo engine */
CFerBind *cairoCFerBind_createWindow(void);
/* The createWindow function for the PyQtCairo engine */
CFerBind *pyqtcairoCFerBind_createWindow(const char *windowname, int windnamelen,
                                         int visible);

#endif
