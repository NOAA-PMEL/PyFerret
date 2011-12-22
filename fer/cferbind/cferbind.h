#ifndef CFERBIND_H_
#define CFERBIND_H_

/* Names of recognized engines */
extern const char *CairoCFerBindName;

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
     int       (*deleteWindow)(struct CFerBind_struct *self);
     int       (*setAntialias)(struct CFerBind_struct *self, int antialias);
     int       (*beginView)(struct CFerBind_struct *self,
                            double leftfrac, double bottomfrac,
                            double rightfrac, double topfrac, int clipit);
     int       (*clipView)(struct CFerBind_struct *self, int clipit);
     int       (*endView)(struct CFerBind_struct *self);
     int       (*updateWindow)(struct CFerBind_struct *self);
     int       (*clearWindow)(struct CFerBind_struct *self, void *fillcolor);
     double*   (*windowDpi)(struct CFerBind_struct *self);
     int       (*resizeWindow)(struct CFerBind_struct *self,
                               double width, double height);
     int       (*showWindow)(struct CFerBind_struct *self, int visible);
     int       (*saveWindow)(struct CFerBind_struct *self,
                             char *filename, int namelen,
                             char *format, int fmtlen, int transbkg);
     void*     (*createColor)(struct CFerBind_struct *self,
                              double redfrac, double greenfrac, double bluefrac);
     int       (*deleteColor)(struct CFerBind_struct *self, void *color);
     void*     (*createFont)(struct CFerBind_struct *self,
                             char *familyname, int namelen, double fontsize,
                             int italic, int bold, int underlined);
     int       (*deleteFont)(struct CFerBind_struct *self, void *font);
     void*     (*createPen)(struct CFerBind_struct *self, void *color,
                            double width, char *style, int stlen,
                            char *capstyle, int capstlen,
                            char *joinstyle, int joinstlen);
     int       (*deletePen)(struct CFerBind_struct *self, void *pen);
     void*     (*createBrush)(struct CFerBind_struct *self,
                              void *color, char *style, int stlen);
     int       (*deleteBrush)(struct CFerBind_struct *self, void *brush);
     void*     (*createSymbol)(struct CFerBind_struct *self,
                               char *symbolname, int namelen);
     int       (*deleteSymbol)(struct CFerBind_struct *self, void *symbol);
     int       (*drawMultiline)(struct CFerBind_struct *self,
                                double ptsx[], double ptsy[], void *pen);
     int       (*drawPoints)(struct CFerBind_struct *self,
                             double ptsx[], double ptsy[], void *symbol,
                             void *color, double ptsize);
     int       (*drawPolygon)(struct CFerBind_struct *self, double ptsx[], 
                              double ptsy[], void *brush, void *pen);
     int       (*drawRectangle)(struct CFerBind_struct *self,
                                double left, double bottom, double right,
                                double top, void *brush, void *pen);
     int       (*drawMulticoloredRectangle)(struct CFerBind_struct *self,
                           double left, double bottom, double right,
                           double top, int numrows, int numcols, void *colors[]);
     int       (*drawText)(struct CFerBind_struct *self, char *text, int textlen,
                           double startx, double starty, void *font, void *color,
                           double rotation);
} CFerBind;

/*
 * Calls the appropriate createWindow function to create the bindings
 * instance and any other appropriate initialization for this "Window". 
 */
CFerBind *cferbind_createWindow(char *enginename, int engnamelen,
                                char *windowname, int winnamelen, int visible);

/* The createWindow function for the Cairo engine */
CFerBind* cairoCFerBind_createWindow(void);

#endif
