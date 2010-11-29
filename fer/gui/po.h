#include <stdio.h>
#include "ferret_structures.h"

/*#include "lines_1.xpm"
#include "lines_1_ins.xpm"
#include "lines_2.xpm"
#include "lines_3.xpm"
#include "lines_4.xpm"
#include "lines_5.xpm"
#include "lines_6.xpm"
#include "sym_1.xpm"
#include "sym_2.xpm"
#include "sym_3.xpm"
#include "sym_4.xpm"
#include "sym_5.xpm"
#include "sym_6.xpm"
#include "sym_7.xpm"
#include "sym_8.xpm"
#include "sym_9.xpm"
#include "sym_10.xpm"
#include "sym_11.xpm"
#include "sym_12.xpm"
#include "sym_13.xpm"
#include "sym_14.xpm"
#include "sym_15.xpm"
#include "sym_16.xpm"
#include "sym_17.xpm"
#include "sym_18.xpm"
#include "sym_19.xpm"
#include "sym_20.xpm"
#include "sym_21.xpm"
#include "sym_22.xpm"
#include "sym_23.xpm"
#include "sym_24.xpm"
#include "sym_25.xpm"
#include "sym_26.xpm"
#include "sym_27.xpm"
#include "sym_28.xpm"
#include "sym_29.xpm"
#include "sym_30.xpm"
#include "sym_31.xpm"
#include "sym_32.xpm"
#include "sym_33.xpm"
#include "sym_34.xpm"
#include "sym_35.xpm"
#include "sym_36.xpm"
#include "sym_37.xpm"
#include "sym_38.xpm"
#include "sym_39.xpm"
#include "sym_40.xpm"
#include "sym_41.xpm"
#include "sym_42.xpm"
#include "sym_43.xpm"
#include "sym_44.xpm"
#include "sym_45.xpm"
#include "sym_46.xpm"
#include "sym_47.xpm"
#include "sym_48.xpm"
#include "sym_49.xpm"
#include "sym_50.xpm"
#include "sym_51.xpm"
#include "sym_52.xpm"
#include "sym_53.xpm"
#include "sym_54.xpm"
#include "sym_55.xpm"
#include "sym_56.xpm"
#include "sym_57.xpm"
#include "sym_58.xpm"
#include "sym_59.xpm"
#include "sym_60.xpm"
#include "sym_61.xpm"
#include "sym_62.xpm"
#include "sym_63.xpm"
#include "sym_64.xpm"
#include "sym_65.xpm"
#include "sym_66.xpm"
#include "sym_67.xpm"
#include "sym_68.xpm"
#include "sym_69.xpm"
#include "sym_70.xpm"
#include "sym_71.xpm"
#include "sym_72.xpm"
#include "sym_73.xpm"
#include "sym_74.xpm"
#include "sym_75.xpm"
#include "sym_76.xpm"
#include "sym_77.xpm"
#include "sym_78.xpm"
#include "sym_79.xpm"
#include "sym_80.xpm"
#include "sym_81.xpm"
#include "sym_82.xpm"
#include "sym_83.xpm"
#include "sym_84.xpm"
#include "sym_85.xpm"
#include "sym_86.xpm"
#include "sym_87.xpm"
#include "sym_88.xpm"*/

swidget create_PlotOptions(swidget UxParent);
static void InitPixmaps();
void SetInitialState(void);
extern Pixmap GetPixmapFromData(char **inData);

/* visual state of plot options */
void UnmapPlotOptions(void);
void Map1DOptions(void);
void Map2DOptions(void);
void MapVectorOptions(void);
void MapVectorSickOptions(void);

/* functions that read state of interface and store */
void Update1DOptionsCB(void);
void UpdateOtherOptionsCB(void);
void Update2DOptionsCB(void);
void UpdateVectorOptionsCB(void);
void UpdateLineStyleCB(Widget wid, XtPointer client_data,
	       XtPointer cbs);
void UpdateLineSymbolCB(Widget wid, XtPointer client_data,
	       XtPointer cbs);

/* external functions */
extern char *FormatFloatStr(double inNum);
extern swidget create_VectorOptions(swidget UxParent);

/* globals */
swidget gSavedPlotOptions = NULL;
swidget PlotOptions;
extern swidget scrolledList1, VectorOptions;
extern int gSelectedContext[100];
extern Context gAllContexts[100];
Widget indAxisButtons[50];
int numIndAxisButtons = 0;
char indAxisVar[32];
extern int gPlotTypeSelected;
static int localPlotType;

static Widget styleWidgets[7] = {lineStyle1_b15, lineStyle1_b9, lineStyle1_b10,
				 lineStyle1_b11, lineStyle1_b12, lineStyle1_b13,
				 lineStyle1_b14};

static Widget symbolWidgets[89] = {lineStyle_b1, optionMenu_p_b10, lineStyle_b89,
				 lineStyle_b91, lineStyle_b92, lineStyle1_b93,
				 lineStyle_b94, lineStyle_b95, lineStyle1_b96,
				 lineStyle_b97, lineStyle_b98, lineStyle1_b99,
				 lineStyle_b100, lineStyle_b101, lineStyle1_b102,
				 lineStyle_b103, lineStyle_b104, lineStyle1_b105,
				 lineStyle_b106, lineStyle_b107, lineStyle1_b108,
				 lineStyle_b109, lineStyle_b110, lineStyle1_b111,
				 lineStyle_b112, lineStyle_b113, lineStyle1_b114,
				 lineStyle_b115, lineStyle_b116, lineStyle1_b117,
				 lineStyle_b118, lineStyle_b119, lineStyle1_b120,
				 lineStyle_b121, lineStyle_b122, lineStyle1_b123,
				 lineStyle_b124, lineStyle_b125, lineStyle1_b126,
				 lineStyle_b127, lineStyle_b128, lineStyle1_b129,
				 lineStyle_b130, lineStyle_b131, lineStyle1_b132,
				 lineStyle_b133, lineStyle_b134, lineStyle1_b135,
				 lineStyle_b136, lineStyle_b137, lineStyle1_b138,
				 lineStyle_b139, lineStyle_b140, lineStyle1_b141,
				 lineStyle_b142, lineStyle_b143, lineStyle1_b144,
				 lineStyle_b145, lineStyle_b146, lineStyle1_b147,
				 lineStyle_b148, lineStyle_b149, lineStyle1_b150,
				 lineStyle_b151, lineStyle_b152, lineStyle1_b153,
				 lineStyle_b154, lineStyle_b155, lineStyle1_b156,
				 lineStyle_b157, lineStyle_b158, lineStyle1_b159,
				 lineStyle_b160, lineStyle_b161, lineStyle1_b162,
				 lineStyle_b163, lineStyle_b164, lineStyle1_b165,
				 lineStyle_b166, lineStyle_b167, lineStyle1_b168,
				 lineStyle_b169, lineStyle_b170, lineStyle1_b171,
				 lineStyle_b172, lineStyle_b173, lineStyle1_b174,
				 lineStyle_b175, lineStyle_b176};