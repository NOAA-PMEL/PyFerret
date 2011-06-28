#include <stdio.h>
#include <Xm/CascadeB.h>
#include "ferret_structures.h"

typedef struct _menu_item {
    char        label[64];         /* the label for the item */
    WidgetClass *class;         /* pushbutton, label, separator... */
    void       (*callback)();   /* routine to call; NULL if none */
    char    callback_data[32]; /* client_data for callback() */
    struct _menu_item *subitems; /* pullright menu items, if not NULL */
} MenuItem;

/* prototypes */
static void InitialHelperState(void);
static void BuildAnomolyVarMenus(void);
swidget create_Helper(swidget UxParent);
static void VarSelectionCB1(Widget UxWidget, XtPointer UxClientData, XtPointer UxCallbackArg);
static void VarSelectionCB2(Widget UxWidget, XtPointer UxClientData, XtPointer UxCallbackArg);
extern void GetDataSets(char dataSetList[][64], int *numDataSets);
extern void GetDSVars(char *dataSetchar, char varList[][32], int *numVars);
extern Widget BuildPulldownMenu(Widget parent, char menu_title[64], Boolean tear_off, 
	MenuItem *items, int height, int width, Boolean theTop);
extern int InsertTextIntoField(char *buffer);
static void FillAveRanges(char *dset, char *var);
static void ReadZTVectorCoords(char *dset, char *var);
static void SetupZTScrollBars(int grp, int i);
static void ScrollBarCB(XtPointer clientData, XtPointer cbArg);
static void ReadAxisCoords(char *dset, char *var, int *hasDepthAxis, int *hasTimeAxis);

/* globals */
swidget gSavedHelper = NULL;
swidget Helper;
static Widget dataSetMenus[2]; /* 2 menus with an arbitrary # of  data set buttons */
static char helperCmnd[512];
int whichHelper;
extern swidget CreateVariable;
extern int gNumDataSets;
static float valVectors[4][500];	/* arrays for use by helpers */
static int valIndices[4][500], numValsOnVector[4];
static Widget helperWidgets[10][10];	/* array of widgets to use as necesary */

