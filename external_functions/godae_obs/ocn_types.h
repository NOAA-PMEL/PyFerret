c
c  CODA Types
c
c  %W% %G%
c
c     ..CODA observation data types
c
c     Name                   Description
c   ---------         ---------------------------
c   MAX_TYPES         maximum number data types
c   data_lbl          data type labels
c
      integer    MAX_TYPES
      parameter (MAX_TYPES = 41)
c
      character data_lbl (0:MAX_TYPES) * 17
c
c   0 = All Data Combined
      data      data_lbl(0)  / 'All Data Combined' /
c   1 = Bathy Temperatures (C)
      data      data_lbl(1)  / '    eXpendable BT' /
c   2 = NOAA14 Day MCSSTs (C)
      data      data_lbl(2)  / '    N14 Day MCSST' /
c   3 = SHIP Engine Room Intake (C)
      data      data_lbl(3)  / '         ERI SHIP' /
c   4 = Fixed BUOY (C)
      data      data_lbl(4)  / '       Fixed BUOY' /
c   5 = Drifting BUOY (C)
      data      data_lbl(5)  / '    Drifting BUOY' /
c   6 = NOAA14 Night MCSSTs (C)
      data      data_lbl(6)  / '  N14 Night MCSST' /
c   7 = NOAA14 Relaxed Day MCSSTs (C)
      data      data_lbl(7)  / 'N14 Rlx Day MCSST' /
c   8 = SSM/I F11 Ice (%)
      data      data_lbl(8)  / '    SSM/I F11 Ice' /
c   9 = SSM/I F13 Ice (%)
      data      data_lbl(9)  / '    SSM/I F13 Ice' /
c  10 = SSM/I F14 Ice (%)
      data      data_lbl(10) / '    SSM/I F14 Ice' /
c  11 = Supplemental Ice (%)
      data      data_lbl(11) / 'ECMWF Ice CLIMATE' /
c  12 = Topex (M)
      data      data_lbl(12) / '        Topex SSH' /
c  13 = ERS2 (M)
      data      data_lbl(13) / '         ERS2 SSH' /
c  14 = GFO (M)
      data      data_lbl(14) / '          GFO SSH' /
c  15 = MODAS Temperature (C)
      data      data_lbl(15) / 'MODAS Temperature' /
c  16 = NCEP SST or GDEM 3D Climatology (C)
      data      data_lbl(16) / ' NCEP SST CLIMATE' /
c  17 = GOES9 Day SSTs (C)
      data      data_lbl(17) / '    GOES9 Day SST' /
c  18 = GOES9 Night SSTs (C)
      data      data_lbl(18) / '  GOES9 Night SST' /
c  19 = GOES9 Relaxed Day SSTs (C)
      data      data_lbl(19) / 'GOES9 Rlx Day SST' /
c  20 = TESAC Temperature (C)
      data      data_lbl(20) / 'TESAC Temperature' /
c  21 = SHIP Bucket (C)
      data      data_lbl(21) / '      Bucket SHIP' /
c  22 = SHIP Hull Sensor (C)
      data      data_lbl(22) / ' Hull Sensor SHIP' /
c  23 = CMAN SST (C)
      data      data_lbl(23) / '         CMAN SST' /
c  24 = NOAA15 Day MCSSTs (C)
      data      data_lbl(24) / '    N15 Day MCSST' /
c  25 = NOAA15 Night MCSSTs (C)
      data      data_lbl(25) / '  N15 Night MCSST' /
c  26 = NOAA15 Relaxed Day MCSSTs (C)
      data      data_lbl(26) / 'N15 Rlx Day MCSST' /
c  27 = Mechanical BT (C)
      data      data_lbl(27) / '    Mechanical BT' /
c  28 = Hydrocast BT (C)
      data      data_lbl(28) / '     Hydrocast BT' /
c  29 = SSM/I F15 Ice (%)
      data      data_lbl(29) / '    SSM/I F15 Ice' /
c  30 = In Situ Sea Surface Height Anomaly (M)
      data      data_lbl(30) / '      In Situ SSH' /
c  31 = SSM/I Ice Shelf
      data      data_lbl(31) / '  SSM/I Ice Shelf' /
c  32 = TESAC Salinity (PSU)
      data      data_lbl(32) / '   TESAC Salinity' /
c  33 = MODAS Salinity (PSU)
      data      data_lbl(33) / '   MODAS Salinity' /
c  34 = TRACK OB Temperature (C)
      data      data_lbl(34) / '    TRACK OB Temp' /
c  35 = TRACK OB Salinty (PSU)
      data      data_lbl(35) / '    TRACK OB Salt' /
c  36 = PALACE Float Temperature (C)
      data      data_lbl(36) / 'PALACE Float Temp' /
c  37 = PALACE Float Salinity (PSU)
      data      data_lbl(37) / 'PALACE Float Salt' /
c  38 = Supplemental MODAS far-field temperature (C)
      data      data_lbl(38) / ' MODAS Suppl Temp' /
c  39 = Supplemental MODAS far-field salinity (PSU)
      data      data_lbl(39) / ' MODAS Suppl Salt' /
c  40 = Supplemental Sea Surface Height Anomaly (M)
      data      data_lbl(40) / ' Supplemental SSH' /
c  41 = Supplemental Sea Ice SSTs (C)
      data      data_lbl(41) / '      Sea Ice SST' /
c
c..End CODA Types
c
