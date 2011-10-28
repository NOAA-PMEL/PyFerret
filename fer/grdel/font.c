/*
 * Font objects can only be used the Window
 * from which they were created.
 */

#include <Python.h> /* make sure Python.h is first */
#include <string.h>
#include "grdel.h"
#include "pyferret.h"

static const char *grdelfontid = "GRDEL_FONT";

typedef struct GDfont_ {
    const char *id;
    grdelType window;
    PyObject *object;
} GDFont;


/*
 * Returns a Font object.
 *
 * Arguments:
 *     window: Window in which this font is to be used
 *     familyname: name of the font family (e.g., "Helvetica", "Times");
 *                 NULL or an empty string uses the default font
 *     familynamelen: actual length of the font family name
 *     fontsize: desired size of the font View units
 *     italic: use the italic version of the font?
 *     bold: use the bold version of the font?
 *     underlined: use the underlined version of the font?
 *
 * Returns a pointer to the font object created.  If an error occurs,
 * NULL is returned and grdelerrmsg contains an explanatory message.
 */
grdelType grdelFont(grdelType window, const char *familyname,
               int familynamelen, float fontsize, grdelBool italic,
               grdelBool bold, grdelBool underlined)
{
    PyObject *bindings;
    PyObject *italicbool;
    PyObject *boldbool;
    PyObject *underlinedbool;
    GDFont *font;

    bindings = grdelWindowVerify(window);
    if ( bindings == NULL ) {
        strcpy(grdelerrmsg, "grdelFont: window argument is not "
                            "a grdel Window");
        return NULL;
    }

    font = (GDFont *) PyMem_Malloc(sizeof(GDFont));
    if ( font == NULL ) {
        strcpy(grdelerrmsg, "grdelFont: out of memory for a new Font");
        return NULL;
    }

    font->id = grdelfontid;
    font->window = window;
    if ( italic )
        italicbool = Py_True;
    else
        italicbool = Py_False;
    if ( bold )
        boldbool = Py_True;
    else
        boldbool = Py_False;
    if ( underlined )
        underlinedbool = Py_True;
    else
        underlinedbool = Py_False;
    font->object = PyObject_CallMethod(bindings, "createFont", "s#dOOO",
                            familyname, familynamelen, (double) fontsize,
                            italicbool, boldbool, underlinedbool);
    if ( font->object == NULL ) {
        sprintf(grdelerrmsg, "grdelFont: error when calling "
                "the binding's createFont method: %s", pyefcn_get_error());
        PyMem_Free(font);
        return NULL;
    }

#ifdef VERBOSEDEBUG
    fprintf(debuglogfile, "grdelFont created: "
            "window = %X, fontsize = %f, font = %X\n",
            window, fontsize, font);
    fflush(debuglogfile);
#endif

    grdelerrmsg[0] = '\0';
    return font;
}

/*
 * Verifies font is a grdel Font.  If window is not NULL,
 * also verifies font can be used with this Window.
 * Returns a pointer to the graphic engine's font object
 * if successful.  Returns NULL if there is a problem.
 */
PyObject *grdelFontVerify(grdelType font, grdelType window)
{
    GDFont *myfont;

    if ( font == NULL )
        return NULL;
    myfont = (GDFont *) font;
    if ( myfont->id != grdelfontid )
        return NULL;
    if ( (window != NULL) && (myfont->window != window) )
        return NULL;
    return myfont->object;
}

/*
 * Delete a Font created by grdelFont
 *
 * Arguments:
 *     font: Font to be deleted
 *
 * Returns success (nonzero) or failure (zero).
 * If failure, grdelerrmsg contains an explanatory message.
 */
grdelBool grdelFontDelete(grdelType font)
{
    GDFont *myfont;
    PyObject *bindings;
    PyObject *result;

#ifdef VERBOSEDEBUG
    fprintf(debuglogfile, "grdelFontDelete called: "
            "font = %X\n", font);
    fflush(debuglogfile);
#endif

    if ( grdelFontVerify(font, NULL) == NULL ) {
        strcpy(grdelerrmsg, "grdelFontDelete: font argument is not "
                            "a grdel Font");
        return (grdelBool) 0;
    }
    myfont = (GDFont *) font;

    bindings = grdelWindowVerify(myfont->window);
    /* "N" - steals the reference to this font object */
    result = PyObject_CallMethod(bindings, "deleteFont", "N",
                                 myfont->object);
    if ( result == NULL )
        sprintf(grdelerrmsg, "grdelFontDelete: error when calling "
                "the binding's deleteFont method: %s", pyefcn_get_error());
    else
        Py_DECREF(result);

    myfont->id = NULL;
    myfont->window = NULL;
    myfont->object = NULL;
    PyMem_Free(font);

    grdelerrmsg[0] = '\0';
    return (grdelBool) 1;
}

/*
 * Creates a Font object.
 *
 * Input Arguments:
 *     window: Window in which this font is to be used
 *     familyname: name of the font family (e.g., "Helvetica", "Times");
 *                 an empty string uses the default font
 *     namelen: actual length of the font family name
 *     fontsize: desired size of the font View units
 *     italic: use the italic version of the font? non-zero yes, zero no.
 *     bold: use the bold version of the font? non-zero yes, zero no.
 *     underlined: use the underlined version of the font? non-zero yes, zero no.
 * Output Arguments:
 *     font: the created font object, or zero if failure.
 *           Use fgderrmsg_ to retrieve the error message.
 */
void fgdfont_(void **font, void **window, char *familyname, int *namelen,
               float *fontsize, int *italic, int *bold, int *underlined)
{
    grdelType myfont;

    myfont = grdelFont(*window, familyname, *namelen, *fontsize,
                       *italic, *bold, *underlined);
    *font = myfont;
}

/*
 * Delete a Font created by fgdfont_
 *
 * Input Arguments:
 *     font: Font to be deleted
 * Output Arguments:
 *     success: non-zero if successful; zero if an error occurred.
 *              Use fgderrmsg_ to retrieve the error message.
 */
void fgdfontdel_(int *success, void **font)
{
    grdelBool result;

    result = grdelFontDelete(*font);
    *success = result;
}

