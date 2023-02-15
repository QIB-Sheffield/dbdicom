# https://github.com/pydicom/pydicom/blob/master/pydicom/_dicom_dict.py
dict = {
    (0x0002, 0x0000): ("File Meta Information Group Length", "UL"),
    (0x0002, 0x0001): ("File Meta Information Version", "OB"),
    }

# (0002, 0000)	File Meta Information Group Length	UL
# (0002, 0001)	File Meta Information Version	OB
# (0002, 0002)	Media Storage SOP Class UID	UI
# (0002, 0003)	Media Storage SOP Instance UID	UI
# (0002, 0010)	Transfer Syntax UID	UI
# (0002, 0012)	Implementation Class UID	UI
# (0002, 0013)	Implementation Version Name	SH
# (0008, 0005)	Specific Character Set	CS
# (0008, 0008)	Image Type	CS
# (0008, 0012)	Instance Creation Date	DA
# (0008, 0013)	Instance Creation Time	TM
# (0008, 0014)	Instance Creator UID	UI
# (0008, 0016)	SOP Class UID	UI
# (0008, 0018)	SOP Instance UID	UI
# (0008, 0020)	Study Date	DA
# (0008, 0021)	Series Date	DA
# (0008, 0022)	Acquisition Date	DA
# (0008, 0023)	Content Date	DA
# (0008, 0030)	Study Time	TM
# (0008, 0031)	Series Time	TM
# (0008, 0032)	Acquisition Time	TM
# (0008, 0033)	Content Time	TM
# (0008, 0060)	Modality	CS
# (0008, 0064)	Conversion Type	CS
# (0008, 0070)	Manufacturer	LO
# (0008, 0080)	Institution Name	LO
# (0008, 0081)	Institution Address	ST
# (0008, 0090)	Referring Physician's Name	PN
# (0008, 0100)	Code Value	SH
# (0008, 0102)	Coding Scheme Designator	SH
# (0008, 0104)	Code Meaning	LO
# (0008, 1010)	Station Name	SH
# (0008, 1030)	Study Description	LO
# (0008, 103e)	Series Description	LO
# (0008, 1040)	Institutional Department Name	LO
# (0008, 1050)	Performing Physician's Name	PN
# (0008, 1070)	Operators' Name	PN
# (0008, 1080)	Admitting Diagnoses Description	LO
# (0008, 1090)	Manufacturer's Model Name	LO
# (0008, 1111)	Referenced Performed Procedure Step Sequence	SQ
#      (0008, 0012)	Instance Creation Date	DA
#      (0008, 0013)	Instance Creation Time	TM
#      (0008, 0014)	Instance Creator UID	UI
#      (0008, 1150)	Referenced SOP Class UID	UI
#      (0008, 1155)	Referenced SOP Instance UID	UI
#      (0020, 0013)	Instance Number	IS
#      (2005, 0014)	Private Creator	LO
#      (2005, 1404)	Private tag data	SS
#      (2005, 1406)	[Unknown]	SS
# (0008, 1140)	Referenced Image Sequence	SQ
#      (0008, 1150)	Referenced SOP Class UID	UI
#      (0008, 1155)	Referenced SOP Instance UID	UI
#      (0008, 1150)	Referenced SOP Class UID	UI
#      (0008, 1155)	Referenced SOP Instance UID	UI
#      (0008, 1150)	Referenced SOP Class UID	UI
#      (0008, 1155)	Referenced SOP Instance UID	UI
# (0010, 0010)	Patient's Name	PN
# (0010, 0020)	Patient ID	LO
# (0010, 0040)	Patient's Sex	CS
# (0010, 1030)	Patient's Weight	DS
# (0010, 4000)	Patient Comments	LT
# (0012, 0063)	De-identification Method	LO
# (0012, 0064)	De-identification Method Code Sequence	SQ
#      (0008, 0100)	Code Value	SH
#      (0008, 0102)	Coding Scheme Designator	SH
#      (0008, 0103)	Coding Scheme Version	SH
#      (0008, 0104)	Code Meaning	LO
#      (0008, 0100)	Code Value	SH
#      (0008, 0102)	Coding Scheme Designator	SH
#      (0008, 0103)	Coding Scheme Version	SH
#      (0008, 0104)	Code Meaning	LO
# (0018, 0010)	Contrast/Bolus Agent	LO
# (0018, 0015)	Body Part Examined	CS
# (0018, 0020)	Scanning Sequence	CS
# (0018, 0021)	Sequence Variant	CS
# (0018, 0022)	Scan Options	CS
# (0018, 0023)	MR Acquisition Type	CS
# (0018, 0050)	Slice Thickness	DS
# (0018, 0080)	Repetition Time	DS
# (0018, 0081)	Echo Time	DS
# (0018, 0083)	Number of Averages	DS
# (0018, 0084)	Imaging Frequency	DS
# (0018, 0085)	Imaged Nucleus	SH
# (0018, 0086)	Echo Number(s)	IS
# (0018, 0087)	Magnetic Field Strength	DS
# (0018, 0088)	Spacing Between Slices	DS
# (0018, 0089)	Number of Phase Encoding Steps	IS
# (0018, 0091)	Echo Train Length	IS
# (0018, 0093)	Percent Sampling	DS
# (0018, 0094)	Percent Phase Field of View	DS
# (0018, 0095)	Pixel Bandwidth	DS
# (0018, 1000)	Device Serial Number	LO
# (0018, 1010)	Secondary Capture Device ID	LO
# (0018, 1016)	Secondary Capture Device Manufacturer	LO
# (0018, 1018)	Secondary Capture Device Manufacturer's Model Name	LO
# (0018, 1019)	Secondary Capture Device Software Versions	LO
# (0018, 1020)	Software Versions	LO
# (0018, 1022)	Video Image Format Acquired	SH
# (0018, 1023)	Digital Image Format Acquired	LO
# (0018, 1030)	Protocol Name	LO
# (0018, 1060)	Trigger Time	DS
# (0018, 1081)	Low R-R Value	IS
# (0018, 1082)	High R-R Value	IS
# (0018, 1083)	Intervals Acquired	IS
# (0018, 1084)	Intervals Rejected	IS
# (0018, 1088)	Heart Rate	IS
# (0018, 1094)	Trigger Window	IS
# (0018, 1100)	Reconstruction Diameter	DS
# (0018, 1250)	Receive Coil Name	SH
# (0018, 1310)	Acquisition Matrix	US
# (0018, 1312)	In-plane Phase Encoding Direction	CS
# (0018, 1314)	Flip Angle	DS
# (0018, 1316)	SAR	DS
# (0018, 1318)	dB/dt	DS
# (0018, 1320)	B1rms	FL
# (0018, 5100)	Patient Position	CS
# (0018, 9073)	Acquisition Duration	FD
# (0018, 9087)	Diffusion b-value	FD
# (0018, 9089)	Diffusion Gradient Orientation	FD
# (0020, 000d)	Study Instance UID	UI
# (0020, 000e)	Series Instance UID	UI
# (0020, 0010)	Study ID	SH
# (0020, 0011)	Series Number	IS
# (0020, 0012)	Acquisition Number	IS
# (0020, 0013)	Instance Number	IS
# (0020, 0032)	Image Position (Patient)	DS
# (0020, 0037)	Image Orientation (Patient)	DS
# (0020, 0052)	Frame of Reference UID	UI
# (0020, 0060)	Laterality	CS
# (0020, 0100)	Temporal Position Identifier	IS
# (0020, 0105)	Number of Temporal Positions	IS
# (0020, 1040)	Position Reference Indicator	LO
# (0020, 1041)	Slice Location	DS
# (0028, 0002)	Samples per Pixel	US
# (0028, 0004)	Photometric Interpretation	CS
# (0028, 0010)	Rows	US
# (0028, 0011)	Columns	US
# (0028, 0030)	Pixel Spacing	DS
# (0028, 0100)	Bits Allocated	US
# (0028, 0101)	Bits Stored	US
# (0028, 0102)	High Bit	US
# (0028, 0103)	Pixel Representation	US
# (0028, 1050)	Window Center	DS
# (0028, 1051)	Window Width	DS
# (0028, 1052)	Rescale Intercept	DS
# (0028, 1053)	Rescale Slope	DS
# (0028, 1054)	Rescale Type	LO
# (0032, 1032)	Requesting Physician	PN
# (0032, 1033)	Requesting Service	LO
# (0032, 1060)	Requested Procedure Description	LO
# (0032, 1070)	Requested Contrast Agent	LO
# (0032, 4000)	Study Comments	LT
# (0038, 0050)	Special Needs	LO
# (0040, 0006)	Scheduled Performing Physician's Name	PN
# (0040, 0241)	Performed Station AE Title	AE
# (0040, 0242)	Performed Station Name	SH
# (0040, 0243)	Performed Location	SH
# (0040, 0244)	Performed Procedure Step Start Date	DA
# (0040, 0245)	Performed Procedure Step Start Time	TM
# (0040, 0250)	Performed Procedure Step End Date	DA
# (0040, 0251)	Performed Procedure Step End Time	TM
# (0040, 0252)	Performed Procedure Step Status	CS
# (0040, 0253)	Performed Procedure Step ID	SH
# (0040, 0254)	Performed Procedure Step Description	LO
# (0040, 0255)	Performed Procedure Type Description	LO
# (0040, 0260)	Performed Protocol Code Sequence	SQ
#      (0008, 0100)	Code Value	SH
#      (0008, 0102)	Coding Scheme Designator	SH
#      (0008, 0104)	Code Meaning	LO
#      (0008, 010b)	Context Group Extension Flag	CS
# (0040, 0280)	Comments on the Performed Procedure Step	ST
# (0040, 1001)	Requested Procedure ID	SH
# (0040, 1002)	Reason for the Requested Procedure	LO
# (0040, 1003)	Requested Procedure Priority	SH
# (0040, 1004)	Patient Transport Arrangements	LO
# (0040, 1005)	Requested Procedure Location	LO
# (0040, 2001)	Reason for the Imaging Service Request	LO
# (0040, 2004)	Issue Date of Imaging Service Request	DA
# (0040, 2005)	Issue Time of Imaging Service Request	TM
# (0040, 2009)	Order Enterer's Location	SH
# (0040, 2010)	Order Callback Phone Number	SH
# (0040, 2400)	Imaging Service Request Comments	LT
# (0040, 9096)	Real World Value Mapping Sequence	SQ
#      (0028, 3003)	LUT Explanation	LO
#      (0040, 08ea)	Measurement Units Code Sequence	SQ
#      (0008, 0100)	Code Value	SH
#      (0008, 0102)	Coding Scheme Designator	SH
#      (0008, 0104)	Code Meaning	LO
#      (0008, 0117)	Context UID	UI
#      >  (0040, 9210)	LUT Label	SH
#      >  (0040, 9211)	Real World Value Last Value Mapped	US
#      >  (0040, 9216)	Real World Value First Value Mapped	US
#      >  (0040, 9224)	Real World Value Intercept	FD
#      >  (0040, 9225)	Real World Value Slope	FD
# (2001, 0010)	Private Creator	LO
# (2001, 0011)	Private Creator	LO
# (2001, 1001)	[Chemical Shift]	FL
# (2001, 1002)	[Chemical Shift Number MR]	IS
# (2001, 1003)	[Diffusion B-Factor]	FL
# (2001, 1004)	[Diffusion Direction]	CS
# (2001, 1006)	[Image Enhanced]	CS
# (2001, 1007)	[Image Type ED ES]	CS
# (2001, 1008)	[Phase Number]	IS
# (2001, 1009)	[Unknown]	FL
# (2001, 100a)	[Slice Number MR]	IS
# (2001, 100b)	[Slice Orientation]	CS
# (2001, 100c)	[Unknown]	CS
# (2001, 100e)	[Unknown]	CS
# (2001, 100f)	[Unknown]	SS
# (2001, 1010)	[Cardiac Sync]	CS
# (2001, 1011)	[Diffusion Echo Time]	FL
# (2001, 1012)	[Dynamic Series]	CS
# (2001, 1013)	[EPI Factor]	SL
# (2001, 1014)	[Number of Echoes]	SL
# (2001, 1015)	[Number of Locations]	SS
# (2001, 1016)	[Number of PC Directions]	SS
# (2001, 1017)	[Number of Phases MR]	SL
# (2001, 1018)	[Number of Slices MR]	SL
# (2001, 1019)	[Partial Matrix Scanned]	CS
# (2001, 101a)	[PC Velocity]	FL
# (2001, 101b)	[Prepulse Delay]	FL
# (2001, 101c)	[Prepulse Type]	CS
# (2001, 101d)	[Reconstruction Number MR]	IS
# (2001, 101e)	[Unknown]	CS
# (2001, 101f)	[Respiration Sync]	CS
# (2001, 1020)	[Scanning Technique Description MR]	LO
# (2001, 1021)	[SPIR]	CS
# (2001, 1022)	[Water Fat Shift]	FL
# (2001, 1023)	[Flip Angle Philips]	DS
# (2001, 1024)	[Interactive]	CS
# (2001, 1025)	[Echo Time Display MR]	SH
# (2001, 105f)	Private tag data	SQ
#      (2001, 0010)	Private Creator	LO
#      (2001, 102d)	[Number of Stack Slices]	SS
#      (2001, 1032)	[Stack Radial Angle]	FL
#      (2001, 1033)	[Stack Radial Axis]	CS
#      (2001, 1035)	[Stack Slice Number]	SS
#      (2001, 1036)	[Stack Type]	CS
#      (2005, 0010)	Private Creator	LO
#      (2005, 0014)	Private Creator	LO
#      (2005, 0015)	Private Creator	LO
#      (2005, 1071)	[Unknown]	FL
#      (2005, 1072)	[Unknown]	FL
#      (2005, 1073)	[Unknown]	FL
#      (2005, 1074)	[Unknown]	FL
#      (2005, 1075)	[Unknown]	FL
#      (2005, 1076)	[Unknown]	FL
#      (2005, 1078)	[Unknown]	FL
#      (2005, 1079)	[Unknown]	FL
#      (2005, 107a)	[Unknown]	FL
#      (2005, 107b)	[Unknown]	CS
#      (2005, 107e)	[Unknown]	FL
#      (2005, 1081)	[Unknown]	CS
#      (2005, 143c)	[Unknown]	FL
#      (2005, 143d)	[Unknown]	FL
#      (2005, 143e)	[Unknown]	FL
#      (2005, 1567)	Private tag data	IS
# (2001, 1060)	[Number of Stacks]	SL
# (2001, 1061)	[Unknown]	CS
# (2001, 1062)	[Unknown]	CS
# (2001, 1063)	[Examination Source]	CS
# (2001, 1077)	[GL TrafoType]	CS
# (2001, 107b)	[Acquisition Number]	IS
# (2001, 1081)	[Number of Dynamic Scans]	IS
# (2001, 1082)	[Echo Train Length]	IS
# (2001, 1083)	[Imaging Frequency]	DS
# (2001, 1084)	[Inversion Time]	DS
# (2001, 1085)	[Magnetic Field Strength]	DS
# (2001, 1086)	[Unknown]	IS
# (2001, 1087)	[Imaged Nucleus]	SH
# (2001, 1088)	[Number of Averages]	DS
# (2001, 1089)	[Phase FOV Percent]	DS
# (2001, 108a)	[Sampling Percent]	DS
# (2001, 108b)	[Unknown]	SH
# (2001, 10c8)	Private tag data	LO
# (2001, 10cc)	[Unknown]	ST
# (2001, 10f1)	[Prospective Motion Correction]	FL
# (2001, 10f2)	[Retrospective Motion Correction]	FL
# (2001, 116b)	Private tag data	LO
# (2005, 0010)	Private Creator	LO
# (2005, 0011)	Private Creator	LO
# (2005, 0012)	Private Creator	LO
# (2005, 0013)	Private Creator	LO
# (2005, 0014)	Private Creator	LO
# (2005, 0015)	Private Creator	LO
# (2005, 0016)	Private Creator	LO
# (2005, 1000)	[Unknown]	FL
# (2005, 1001)	[Unknown]	FL
# (2005, 1002)	[Unknown]	FL
# (2005, 1008)	[Unknown]	FL
# (2005, 1009)	[Unknown]	FL
# (2005, 100a)	[Unknown]	FL
# (2005, 100b)	[Unknown]	FL
# (2005, 100c)	[Unknown]	FL
# (2005, 100d)	[Unknown]	FL
# (2005, 100e)	[Unknown]	FL
# (2005, 100f)	[Window Center]	DS
# (2005, 1010)	[Window Width]	DS
# (2005, 1011)	[Unknown]	CS
# (2005, 1012)	[Unknown]	CS
# (2005, 1013)	[Unknown]	CS
# (2005, 1014)	[Unknown]	CS
# (2005, 1015)	[Unknown]	CS
# (2005, 1016)	[Unknown]	CS
# (2005, 1017)	[Unknown]	CS
# (2005, 1018)	[Unknown]	LO
# (2005, 1019)	[Unknown]	CS
# (2005, 101a)	[Unknown]	SS
# (2005, 101b)	[Unknown]	CS
# (2005, 101c)	[Unknown]	CS
# (2005, 101d)	[Unknown]	SS
# (2005, 101e)	[Unknown]	SH
# (2005, 101f)	[Unknown]	SH
# (2005, 1020)	[Number of Chemical Shift]	SL
# (2005, 1021)	[Unknown]	SS
# (2005, 1022)	[Unknown]	IS
# (2005, 1023)	[Unknown]	SS
# (2005, 1025)	[Unknown]	SS
# (2005, 1026)	[Unknown]	CS
# (2005, 1027)	[Unknown]	CS
# (2005, 1028)	[Unknown]	CS
# (2005, 1029)	[Unknown]	CS
# (2005, 102a)	[Unknown]	IS
# (2005, 102b)	[Unknown]	SS
# (2005, 102c)	[Unknown]	CS
# (2005, 102d)	[Unknown]	IS
# (2005, 102e)	[Unknown]	CS
# (2005, 102f)	[Unknown]	CS
# (2005, 1030)	[Repetition Time]	FL
# (2005, 1031)	[Unknown]	CS
# (2005, 1032)	[Unknown]	CS
# (2005, 1033)	[Acquisition Duration]	FL
# (2005, 1034)	[Unknown]	CS
# (2005, 1035)	[Unknown]	CS
# (2005, 1036)	[Unknown]	CS
# (2005, 1037)	[Unknown]	CS
# (2005, 1038)	[Unknown]	CS
# (2005, 1039)	[Unknown]	CS
# (2005, 103a)	[Unknown]	SH
# (2005, 103b)	[Unknown]	CS
# (2005, 103c)	[Unknown]	CS
# (2005, 103d)	[Unknown]	SS
# (2005, 103e)	[Unknown]	SL
# (2005, 105f)	[Unknown]	CS
# (2005, 1060)	[Unknown]	IS
# (2005, 1061)	[Unknown]	CS
# (2005, 1063)	[Unknown]	SS
# (2005, 106e)	[Unknown]	CS
# (2005, 106f)	[Unknown]	CS
# (2005, 1070)	[Unknown]	LO
# (2005, 1081)	[Unknown]	CS
# (2005, 1085)	Private tag data	SQ
#      (2005, 0010)	Private Creator	LO
#      (2005, 1054)	[Unknown]	FL
#      (2005, 1055)	[Unknown]	FL
#      (2005, 1056)	[Unknown]	FL
#      (2005, 1057)	[Unknown]	FL
#      (2005, 1058)	[Unknown]	FL
#      (2005, 1059)	[Unknown]	FL
#      (2005, 105a)	[Unknown]	FL
#      (2005, 105b)	[Unknown]	FL
#      (2005, 105c)	[Unknown]	FL
#      (2005, 105d)	[Unknown]	CS
#      (2005, 105e)	[Unknown]	CS
# (2005, 1086)	[Unknown]	SS
# (2005, 109f)	[Unknown]	CS
# (2005, 10a0)	[Unknown]	FL
# (2005, 10a1)	[Syncra Scan Type]	CS
# (2005, 10a2)	[Unknown]	CS
# (2005, 10a8)	[Unknown]	DS
# (2005, 10a9)	[Unknown]	CS
# (2005, 10b0)	[Diffusion Direction RL]	FL
# (2005, 10b1)	[Diffusion Direction AP]	FL
# (2005, 10b2)	[Diffusion Direction FH]	FL
# (2005, 10c0)	[Unknown]	CS
# (2005, 1134)	Private tag data	LT
# (2005, 1199)	[Unknown]	UL
# (2005, 1200)	[Unknown]	UL
# (2005, 1201)	[Unknown]	UL
# (2005, 1213)	[Unknown]	UL
# (2005, 1245)	[Unknown]	SS
# (2005, 1249)	[Unknown]	SS
# (2005, 1251)	[Unknown]	SS
# (2005, 1252)	[Unknown]	SS
# (2005, 1253)	[Unknown]	SS
# (2005, 1256)	[Unknown]	SS
# (2005, 1325)	[Unknown]	CS
# (2005, 1326)	[Unknown]	FL
# (2005, 1327)	[Unknown]	CS
# (2005, 1328)	[Unknown]	CS
# (2005, 1329)	[Unknown]	FL
# (2005, 1331)	[Unknown]	SS
# (2005, 1334)	[Unknown]	CS
# (2005, 1335)	[Unknown]	CS
# (2005, 1336)	[Unknown]	FL
# (2005, 1337)	[Unknown]	FL
# (2005, 1338)	[Unknown]	FL
# (2005, 1340)	[Unknown]	CS
# (2005, 1341)	[Unknown]	CS
# (2005, 1342)	[Unknown]	CS
# (2005, 1343)	[Unknown]	CS
# (2005, 1345)	[Unknown]	CS
# (2005, 1346)	[Unknown]	CS
# (2005, 1347)	[Unknown]	FL
# (2005, 1348)	[Unknown]	CS
# (2005, 1349)	[Unknown]	FL
# (2005, 1351)	[Unknown]	SS
# (2005, 1352)	[Unknown]	SS
# (2005, 1354)	[Unknown]	CS
# (2005, 1355)	[Unknown]	FL
# (2005, 1356)	[Unknown]	CS
# (2005, 1357)	[Unknown]	SS
# (2005, 1358)	[Unknown]	LO
# (2005, 1359)	[Unknown]	FL
# (2005, 1360)	[Unknown]	FL
# (2005, 1362)	[Unknown]	FL
# (2005, 1363)	[Unknown]	FL
# (2005, 1364)	[Unknown]	CS
# (2005, 1370)	[Unknown]	SS
# (2005, 1381)	[Unknown]	IS
# (2005, 1382)	[Unknown]	UL
# (2005, 1391)	[Unknown]	PN
# (2005, 1392)	[Unknown]	IS
# (2005, 1393)	[Unknown]	IS
# (2005, 1395)	[Unknown]	ST
# (2005, 1396)	[Unknown]	CS
# (2005, 1397)	[Unknown]	LO
# (2005, 1398)	[Unknown]	CS
# (2005, 1399)	[Unknown]	CS
# (2005, 1400)	[Unknown]	CS
# (2005, 1401)	[Unknown]	UL
# (2005, 1403)	[Unknown]	UL
# (2005, 1409)	[Unknown]	DS
# (2005, 140a)	[Unknown]	DS
# (2005, 140b)	[Unknown]	LO
# (2005, 140f)	Private tag data	SQ
#      (0008, 002a)	Acquisition DateTime	DT
#      (0008, 9123)	Creator-Version UID	UI
#      (0008, 9205)	Pixel Presentation	CS
#      (0008, 9206)	Volumetric Properties	CS
#      (0008, 9207)	Volume Based Calculation Technique	CS
#      (0008, 9209)	Acquisition Contrast	CS
#      (0018, 9005)	Pulse Sequence Name	SH
#      (0018, 9008)	Echo Pulse Sequence	CS
#      (0018, 9009)	Inversion Recovery	CS
#      (0018, 9011)	Multiple Spin Echo	CS
#      (0018, 9012)	Multi-planar Excitation	CS
#      (0018, 9014)	Phase Contrast	CS
#      (0018, 9015)	Time of Flight Contrast	CS
#      (0018, 9016)	Spoiling	CS
#      (0018, 9017)	Steady State Pulse Sequence	CS
#      (0018, 9018)	Echo Planar Pulse Sequence	CS
#      (0018, 9019)	Tag Angle First Axis	FD
#      (0018, 9020)	Magnetization Transfer	CS
#      (0018, 9021)	T2 Preparation	CS
#      (0018, 9022)	Blood Signal Nulling	CS
#      (0018, 9024)	Saturation Recovery	CS
#      (0018, 9025)	Spectrally Selected Suppression	CS
#      (0018, 9026)	Spectrally Selected Excitation	CS
#      (0018, 9027)	Spatial Pre-saturation	CS
#      (0018, 9028)	Tagging	CS
#      (0018, 9029)	Oversampling Phase	CS
#      (0018, 9030)	Tag Spacing First Dimension	FD
#      (0018, 9032)	Geometry of k-Space Traversal	CS
#      (0018, 9033)	Segmented k-Space Traversal	CS
#      (0018, 9034)	Rectilinear Phase Encode Reordering	CS
#      (0018, 9035)	Tag Thickness	FD
#      (0018, 9036)	Partial Fourier Direction	CS
#      (0018, 9037)	Cardiac Synchronization Technique	CS
#      (0018, 9043)	Receive Coil Type	CS
#      (0018, 9044)	Quadrature Receive Coil	CS
#      (0018, 9047)	Multi-Coil Element Name	SH
#      (0018, 9050)	Transmit Coil Manufacturer Name	LO
#      (0018, 9051)	Transmit Coil Type	CS
#      (0018, 9053)	Chemical Shift Reference	FD
#      (0018, 9058)	MR Acquisition Frequency Encoding Steps	US
#      (0018, 9059)	De-coupling	CS
#      (0018, 9060)	De-coupled Nucleus	CS
#      (0018, 9062)	De-coupling Method	CS
#      (0018, 9064)	k-space Filtering	CS
#      (0018, 9065)	Time Domain Filtering	CS
#      (0018, 9069)	Parallel Reduction Factor In-plane	FD
#      (0018, 9075)	Diffusion Directionality	CS
#      (0018, 9077)	Parallel Acquisition	CS
#      (0018, 9078)	Parallel Acquisition Technique	CS
#      (0018, 9079)	Inversion Times	FD
#      (0018, 9080)	Metabolite Map Description	ST
#      (0018, 9081)	Partial Fourier	CS
#      (0018, 9085)	Cardiac Signal Source	CS
#      (0018, 9090)	Velocity Encoding Direction	FD
#      (0018, 9091)	Velocity Encoding Minimum Value	FD
#      (0018, 9093)	Number of k-Space Trajectories	US
#      (0018, 9094)	Coverage of k-Space	CS
#      (0018, 9101)	Frequency Correction	CS
#      (0018, 9147)	Diffusion Anisotropy Type	CS
#      (0018, 9155)	Parallel Reduction Factor out-of-plane	FD
#      (0018, 9168)	Parallel Reduction Factor Second In-plane	FD
#      (0018, 9169)	Cardiac Beat Rejection Technique	CS
#      (0018, 9170)	Respiratory Motion Compensation Technique	CS
#      (0018, 9171)	Respiratory Signal Source	CS
#      (0018, 9172)	Bulk Motion Compensation Technique	CS
#      (0018, 9174)	Applicable Safety Standard Agency	CS
#      (0018, 9176)	Operating Mode Sequence	SQ
#      (0018, 9177)	Operating Mode Type	CS
#      (0018, 9178)	Operating Mode	CS
#      (0018, 9177)	Operating Mode Type	CS
#      (0018, 9178)	Operating Mode	CS
#      (0018, 9177)	Operating Mode Type	CS
#      (0018, 9178)	Operating Mode	CS
#      >  (0018, 9179)	Specific Absorption Rate Definition	CS
#      >  (0018, 9180)	Gradient Output Type	CS
#      >  (0018, 9181)	Specific Absorption Rate Value	FD
#      >  (0018, 9182)	Gradient Output	FD
#      >  (0018, 9183)	Flow Compensation Direction	CS
#      >  (0018, 9199)	Water Referenced Phase Correction	CS
#      >  (0018, 9200)	MR Spectroscopy Acquisition Type	CS
#      >  (0018, 9231)	MR Acquisition Phase Encoding Steps in-plane	US
#      >  (0018, 9232)	MR Acquisition Phase Encoding Steps out-of-plane	US
#      >  (0018, 9240)	RF Echo Train Length	US
#      >  (0018, 9241)	Gradient Echo Train Length	US
#      >  (0018, 9602)	Diffusion b-value XX	FD
#      >  (0018, 9603)	Diffusion b-value XY	FD
#      >  (0018, 9604)	Diffusion b-value XZ	FD
#      >  (0018, 9605)	Diffusion b-value YY	FD
#      >  (0018, 9606)	Diffusion b-value YZ	FD
#      >  (0018, 9607)	Diffusion b-value ZZ	FD
#      >  (0020, 9072)	Frame Laterality	CS
#      >  (0020, 9254)	Respiratory Interval Time	FD
#      >  (0020, 9255)	Nominal Respiratory Trigger Delay Time	FD
#      >  (0020, 9256)	Respiratory Trigger Delay Threshold	FD
#      >  (0028, 9001)	Data Point Rows	UL
#      >  (0028, 9002)	Data Point Columns	UL
#      >  (0028, 9003)	Signal Domain Columns	CS
#      >  (0028, 9108)	Data Representation	CS
#      >  (0040, 9210)	LUT Label	SH
# (2005, 1410)	[Unknown]	IS
# (2005, 1412)	[Unknown]	IS
# (2005, 1413)	[Unknown]	IS
# (2005, 1414)	[Unknown]	SL
# (2005, 1415)	[Unknown]	SL
# (2005, 1416)	[Unknown]	CS
# (2005, 1418)	[Unknown]	CS
# (2005, 1419)	[Unknown]	CS
# (2005, 141a)	[Unknown]	CS
# (2005, 141b)	[Unknown]	IS
# (2005, 141c)	[Unknown]	IS
# (2005, 141d)	[Unknown]	IS
# (2005, 1426)	[Unknown]	CS
# (2005, 1428)	[Unknown]	SL
# (2005, 1429)	Private tag data	CS
# (2005, 142a)	[Unknown]	CS
# (2005, 142b)	[Unknown]	CS
# (2005, 142c)	[Unknown]	CS
# (2005, 142d)	[Unknown]	CS
# (2005, 142e)	[Unknown]	FL
# (2005, 142f)	[Unknown]	FL
# (2005, 1430)	[Unknown]	FL
# (2005, 1431)	[Unknown]	FL
# (2005, 1432)	[Unknown]	CS
# (2005, 1435)	[Unknown]	CS
# (2005, 1437)	[Unknown]	CS
# (2005, 143a)	[Unknown]	LT
# (2005, 143b)	[Unknown]	CS
# (2005, 143f)	[Unknown]	CS
# (2005, 1440)	[Unknown]	FL
# (2005, 1441)	[Unknown]	FL
# (2005, 1442)	[Unknown]	FL
# (2005, 1443)	[Unknown]	FL
# (2005, 1444)	[Unknown]	IS
# (2005, 1445)	[Unknown]	CS
# (2005, 1446)	[Unknown]	FL
# (2005, 1447)	[Unknown]	FL
# (2005, 1448)	[Unknown]	FL
# (2005, 144d)	[Unknown]	CS
# (2005, 144e)	[Unknown]	CS
# (2005, 1492)	Private tag data	FL
# (2005, 1553)	Private tag data	FL
# (2005, 1554)	Private tag data	FL
# (2005, 1555)	Private tag data	FL
# (2005, 1556)	Private tag data	FL
# (2005, 1557)	Private tag data	CS
# (2005, 1558)	Private tag data	FL
# (2005, 1559)	Private tag data	FL
# (2005, 1560)	Private tag data	IS
# (2005, 1561)	Private tag data	FL
# (2005, 1562)	Private tag data	LT
# (2005, 1563)	Private tag data	CS
# (2005, 1564)	Private tag data	CS
# (2005, 1565)	Private tag data	CS
# (2005, 1566)	Private tag data	CS
# (2005, 1568)	Private tag data	IS
# (2005, 1571)	Private tag data	IS
# (2005, 1572)	Private tag data	FL
# (2005, 1573)	Private tag data	IS
# (2005, 1574)	Private tag data	DS
# (2005, 1575)	Private tag data	DS
# (2005, 1576)	Private tag data	LT
# (2005, 1578)	Private tag data	CS
# (2005, 1579)	Private tag data	CS
# (2005, 1581)	Private tag data	CS
# (2005, 1582)	Private tag data	IS
# (2005, 1583)	Private tag data	LT
# (2005, 1585)	Private tag data	DS
# (2005, 1586)	Private tag data	LT
# (2005, 1587)	Private tag data	DS
# (2005, 1595)	Private tag data	CS
# (2005, 1596)	Private tag data	IS
# (2005, 1597)	Private tag data	CS
# (2005, 1599)	Private tag data	SL
# (2005, 1600)	Private tag data	CS
# (2005, 1601)	Private tag data	FL
# (2005, 1602)	Private tag data	FL
# (2005, 1603)	Private tag data	FL
# (2050, 0020)	Presentation LUT Shape	CS
