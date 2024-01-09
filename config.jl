# details about the sensors:
# https://www.waterprobes.com/sensor-parameters-water-quality-monitoring


# dictionary of targets
# symbol => [units, long-name, min-val]
targets_dict = Dict(
    :Br => ["mg/l", "Br⁻", 0.0],
    :CDOM => ["ppb", "CDOM", 0.0],
    :CO => ["ppb", "Crude Oil", 0.0],
    :Ca => ["mg/l", "Ca⁺⁺", 0.0],
    :Chl => ["μg/l", "Chlorophyll A", 0.0],
    :ChlRed => ["μg/l", "Chlorophyll A\nwith Red Excitation", 0.0],
    :Cl => ["mg/l", "Cl⁻", 0.0],
    :HDO => ["mg/l", "Dissolved Oxygen", 0.0],
    :HDO_percent => ["% Sat", "Dissolved Oxygen", 0.0],
    :NH4 => ["mg/l-N", "NH₄⁺", 0.0],
    :NO3 => ["mg/l-N", "NO₃⁻", 0.0],
    :Na => ["mg/l", "Na⁺", 0.0],
    :OB => ["ppb", "Optical Brighteners", 0.0],
    :RefFuel => ["ppb", "Refined Fuels", 0.0],
    :Salinity3488 => ["PSU", "Salinity"],
    #:Salinity3489 => ["PSU", "Salinity (3489)"],
    :Salinity3490 => ["PSU", "Salinity"],
    :SpCond => ["μS/cm", "Conductivity"],
    :TDS => ["mg/l", "Total Dissolved Solids", 0.0],
    :TRYP => ["ppb", "Tryptophan", 0.0],
    :Temp3488 => ["°C", "Temperature"],
    :Temp3489 => ["°C", "Temperature"],
    :Temp3490 => ["°C", "Temperature"],
    :Turb3489 => ["FNU", "Turbidity"],
    :Turb3488 => ["FNU", "Turbidity"],
    :Turb3490 => ["FNU", "Turbidity"],
    :bg => ["ppb", "Blue-Green Algae\n (Phycocyanin)", 0.0],
    :bgm => ["ppb", "Blue-Green Algae\n (Phycoerythrin)", 0.0],
    :pH => ["", "pH", 0.0],
    :pH_mV => ["mV", "pH"],
)



using LaTeXStrings

targets_latex= Dict(
    :Br => L"$\mathrm{Br}^-$ (mg/l)",
    :CDOM => "CDOM (ppb)",
    :CO => "Crude Oil (ppb)",
    :Ca => L"$\mathrm{Ca}^{2+}$ (mg/l)",
    :Chl => L"Chlorophyll A ($\mu$g/l)",
    :ChlRed => L"Chlorophyll A with Red Excitation ($\mu$g/l)",
    :Cl => L"$\mathrm{Cl^-}$ (mg/l)",
    :HDO => "Dissolved Oxygen (mg/l)",
    :HDO_percent => "Dissolved Oxygen (\\% Sat)",
    :NH4 => L"\mathrm{NH_4^+} (mg/l-N)",
    :NO3 => L"\mathrm{NO_3^-} (mg/l-N)",
    :Na => L"\mathrm{Na^+} (mg/l)",
    :OB => "Optical Brighteners (ppb)",
    :RefFuel => "Refined Fuels (ppb)",
    :Salinity3488 => "Salinity (PSU)",
    :Salinity3490 => "Salinity (PSU)",
    :SpCond => L"Conductivity ($\mu$S/cm)",
    :TDS => "Total Dissolved Solids (mg/l)",
    :TRYP => "Tryptophan (ppb)",
    :Temp3488 => "Temperature (°C)",
    :Temp3489 => "Temperature (°C)",
    :Temp3490 => "Temperature (°C)",
    :Turb3489 => "Turbidity (FNU)",
    :Turb3488 => "Turbidity (FNU)",
    :Turb3490 => "Turbidity (FNU)",
    :bg => "Blue-Green Algae [Phycocyanin] (ppb)",
    :bgm => "Blue-Green Algae [Phycoerythrin] (ppb)",
    :pH => "pH",
    :pH_mV => "pH (mV)",
)





# wavelength bins used by sensor
wavelengths = [390.960, 392.270, 393.590, 394.910, 396.230, 397.550, 398.870, 400.180, 401.500, 402.820, 404.140, 405.460, 406.780, 408.100, 409.420, 410.740, 412.060, 413.380, 414.700, 416.020, 417.340, 418.660, 419.980, 421.300, 422.620, 423.940, 425.260, 426.580, 427.910, 429.230, 430.550, 431.870, 433.190, 434.510, 435.840, 437.160, 438.480, 439.800, 441.120, 442.450, 443.770, 445.090, 446.410, 447.740, 449.060, 450.380, 451.710, 453.030, 454.350, 455.680, 457.000, 458.320, 459.650, 460.970, 462.300, 463.620, 464.940, 466.270, 467.590, 468.920, 470.240, 471.570, 472.890, 474.220, 475.540, 476.870, 478.190, 479.520, 480.840, 482.170, 483.500, 484.820, 486.150, 487.480, 488.800, 490.130, 491.450, 492.780, 494.110, 495.430, 496.760, 498.090, 499.420, 500.740, 502.070, 503.400, 504.730, 506.050, 507.380, 508.710, 510.040, 511.370, 512.700, 514.020, 515.350, 516.680, 518.010, 519.340, 520.670, 522.000, 523.330, 524.660, 525.990, 527.310, 528.640, 529.970, 531.300, 532.630, 533.960, 535.300, 536.630, 537.960, 539.290, 540.620, 541.950, 543.280, 544.610, 545.940, 547.270, 548.600, 549.940, 551.270, 552.600, 553.930, 555.260, 556.600, 557.930, 559.260, 560.590, 561.920, 563.260, 564.590, 565.920, 567.260, 568.590, 569.920, 571.260, 572.590, 573.920, 575.260, 576.590, 577.920, 579.260, 580.590, 581.930, 583.260, 584.600, 585.930, 587.260, 588.600, 589.930, 591.270, 592.600, 593.940, 595.280, 596.610, 597.950, 599.280, 600.620, 601.950, 603.290, 604.630, 605.960, 607.300, 608.640, 609.970, 611.310, 612.650, 613.980, 615.320, 616.660, 617.990, 619.330, 620.670, 622.010, 623.340, 624.680, 626.020, 627.360, 628.700, 630.040, 631.370, 632.710, 634.050, 635.390, 636.730, 638.070, 639.410, 640.750, 642.090, 643.420, 644.760, 646.100, 647.440, 648.780, 650.120, 651.460, 652.800, 654.140, 655.480, 656.830, 658.170, 659.510, 660.850, 662.190, 663.530, 664.870, 666.210, 667.550, 668.900, 670.240, 671.580, 672.920, 674.260, 675.600, 676.950, 678.290, 679.630, 680.970, 682.320, 683.660, 685.000, 686.350, 687.690, 689.030, 690.380, 691.720, 693.060, 694.410, 695.750, 697.090, 698.440, 699.780, 701.130, 702.470, 703.820, 705.160, 706.510, 707.850, 709.200, 710.540, 711.890, 713.230, 714.580, 715.920, 717.270, 718.610, 719.960, 721.310, 722.650, 724.000, 725.340, 726.690, 728.040, 729.380, 730.730, 732.080, 733.420, 734.770, 736.120, 737.470, 738.810, 740.160, 741.510, 742.860, 744.200, 745.550, 746.900, 748.250, 749.600, 750.950, 752.290, 753.640, 754.990, 756.340, 757.690, 759.040, 760.390, 761.740, 763.090, 764.440, 765.790, 767.140, 768.490, 769.840, 771.190, 772.540, 773.890, 775.240, 776.590, 777.940, 779.290, 780.640, 781.990, 783.340, 784.690, 786.050, 787.400, 788.750, 790.100, 791.450, 792.800, 794.160, 795.510, 796.860, 798.210, 799.570, 800.920, 802.270, 803.620, 804.980, 806.330, 807.680, 809.040, 810.390, 811.740, 813.100, 814.450, 815.800, 817.160, 818.510, 819.870, 821.220, 822.580, 823.930, 825.280, 826.640, 827.990, 829.350, 830.700, 832.060, 833.420, 834.770, 836.130, 837.480, 838.840, 840.190, 841.550, 842.910, 844.260, 845.620, 846.980, 848.330, 849.690, 851.050, 852.400, 853.760, 855.120, 856.470, 857.830, 859.190, 860.550, 861.900, 863.260, 864.620, 865.980, 867.340, 868.690, 870.050, 871.410, 872.770, 874.130, 875.490, 876.850, 878.210, 879.560, 880.920, 882.280, 883.640, 885.000, 886.360, 887.720, 889.080, 890.440, 891.800, 893.160, 894.520, 895.880, 897.240, 898.600, 899.970, 901.330, 902.690, 904.050, 905.410, 906.770, 908.130, 909.490, 910.860, 912.220, 913.580, 914.940, 916.300, 917.670, 919.030, 920.390, 921.750, 923.120, 924.480, 925.840, 927.210, 928.570, 929.930, 931.300, 932.660, 934.020, 935.390, 936.750, 938.120, 939.480, 940.840, 942.210, 943.570, 944.940, 946.300, 947.670, 949.030, 950.400, 951.760, 953.130, 954.490, 955.860, 957.220, 958.590, 959.960, 961.320, 962.690, 964.050, 965.420, 966.790, 968.150, 969.520, 970.890, 972.250, 973.620, 974.990, 976.350, 977.720, 979.090, 980.460, 981.820, 983.190, 984.560, 985.930, 987.300, 988.660, 990.030, 991.400, 992.770, 994.140, 995.510, 996.880, 998.250, 999.610, 1000.980, 1002.350, 1003.720, 1005.090, 1006.460, 1007.830, 1009.200, 1010.570]


name_replacements = Dict()
for i ∈ 1:size(wavelengths, 1)
    key = "R_" * lpad(i,3,"0")
    name_replacements[key] = "Reflectance at $(wavelengths[i]) nm"
end

# orientation
name_replacements["roll"] = "Roll"
name_replacements["pitch"] = "Pitch"
name_replacements["heading"] = "Heading"
name_replacements["altitude"] = "Altitude"

# angle to pixel in scanline from center
name_replacements["view_angle"] = "Viewing Angle"

# solar geometry
name_replacements["solar_azimuth"] = "Solar Azimuth"
name_replacements["solar_elevation"] = "Solar Elevation"
name_replacements["solar_zenith"] = "Solar Zenith"

# add derived metrics

name_replacements["DVI"] = "Difference Vegetation Index"
name_replacements["EVI"] = "Enhanced Vegetation Index"
name_replacements["GEMI"] = "Global Environmental Monitoring Index"
name_replacements["GARI"] = "Green Atmospherically Resistant Index"
name_replacements["GCI"] = "Green Chlorophyll Index"
name_replacements["GDVI"] = "Green Difference Vegetation Index"
name_replacements["GLI"] = "Green Leaf Index"
name_replacements["GNDVI"] = "Green Normalized Difference Vegetation Index"
name_replacements["GOSAVI"] = "Green Optimized Soil Adjusted Vegetation Index"
name_replacements["GRVI"] = "Green Ratio Vegetation Index"
name_replacements["GSAVI"] = "Green Soil Adjusted Vegetation Index"
name_replacements["IPVI"] = "Infrared Percentage Vegetation Index"
name_replacements["LAI"] = "Leaf Area Index"
name_replacements["LCI"] = "Leaf Chlorphyll Index"
name_replacements["MNLI"] = "Modified Non-Linear Index"
name_replacements["MSAVI2"] = "Modified Soil Adjusted Vegetation Index 2"
name_replacements["MSR"] = "Modified Simple Ratio"
name_replacements["NLI"] = "Non-Linear Index"
name_replacements["NDVI"] = "Normalized Difference Vegetation Index"
name_replacements["NPCI"] = "Normalized Pigment Chlorophyll Index"
name_replacements["OSAVI"] = "Optimized Soil Ajusted Vegetation Index"
name_replacements["RDVI"] = "Renormalized Difference Vegetation Index"
name_replacements["SR"] = "Simple Ratio"
name_replacements["TDVI"] = "Transformed Difference Vegetation Index"
name_replacements["TGI"] = "Triangular Greenness Index"
name_replacements["VARI"] = "Visible Atmospherically Resistant Index"
name_replacements["WDRVI"] = "Wide Dynamic Range Vegetation Index"
name_replacements["AVRI"] = "Atmospherically Resistant Vegetation Index"
name_replacements["MCARI"] = "Modified Chlorophyll Absorption Ratio Index"
name_replacements["MCARI2"] = "Modified Chlorophyll Absorption Ratio Index Improved"
name_replacements["MRENDVI"] = "Modified Red Edge Normalized Difference Vegetation Index"
name_replacements["MRESER"] = "Modified Red Edge Simple Ratio"
name_replacements["MTVI"] = "Modified Triangular Vegetation Index"
name_replacements["RENDVI"] = "Red Edge Normalized Difference Vegetation Index"
name_replacements["TCARI"] = "Transformed Chlorophyll Absorption Reflectance Index"
name_replacements["TVI"] = "Triangular Vegetation Index"
name_replacements["VREI1"] = "Vogelmann Red Edge Index 1"
name_replacements["VREI2"] = "Vogelmann Red Edge Index 2"
name_replacements["VREI3"] = "Vogelmann Red Edge Index 3"
name_replacements["PRI"] = "Photochemical Reflectance Index"
name_replacements["SIPI"] = "Structure Insensitive Pigment Index"
name_replacements["SIPI1"] = "Structure Independent Pigment Index"
name_replacements["PSRI"] = "Plant Senescence Reflectance Index"
name_replacements["ARI1"] = "Anthocyanin Reflectance Index 1"
name_replacements["ARI2"] = "Anthocyanin Reflectance Index 2"
name_replacements["CRI1"] = "Carotenoid Reflectance Index 1"
name_replacements["CRI2"] = "Carotenoid Reflectance Index 2"
name_replacements["NDWI1"] = "Normalized Difference Water Index 1"
name_replacements["NDWI2"] = "Normalized Difference Water Index 2"
name_replacements["MNDWI"] = "Modified Normalized Difference Water Index"
name_replacements["WBI"] = "Water Band Index"
name_replacements["ACI"] = "Anthocyanin Content Index"
name_replacements["MARI"] = "Modified Anthocyanin Reflectance Index"
name_replacements["MSI"] = "Moisture Stress Index"
name_replacements["MTCI"] = "MERIS Terrestrial Chlorophyll Index"
name_replacements["NDII"] = "Normalized Difference Infrared Index"
name_replacements["NDRE"] = "Normalized Difference Red Edge"
name_replacements["RGRI"] = "Red Green Ratio Index"
name_replacements["RVSI"] = "Red Edge Vegetation Stress Index"

name_replacements["yaw_minus_azimuth"] = "Heading - Solar Azimuth"
name_replacements["Σrad"] = "Total Pixel Intensity"
name_replacements["Σdownwelling"] = "Total Downwelling Intensity"






bad_hsi_dict = Dict(
    "11-23" => Dict(
        "Scotty_1" => [
            "Scotty_1-16.h5",
            "Scotty_1-25.h5",
            "Scotty_1-26.h5",
            "Scotty_1-27.h5",
            "Scotty_1-28.h5",
            "Scotty_1-29.h5",
            "Scotty_1-30.h5",
            "Scotty_1-31.h5",
            "Scotty_1-32.h5",
        ],
        "Scotty_2" => [
            "Scotty_2-9.h5",
            "Scotty_2-10.h5",
            "Scotty_2-16.h5",
            "Scotty_2-20.h5",
            "Scotty_2-21.h5",
            "Scotty_2-22.h5",
            "Scotty_2-23.h5",
            "Scotty_2-25.h5",
            "Scotty_2-26.h5",
            "Scotty_2-27.h5",
            "Scotty_2-28.h5",
            "Scotty_2-29.h5",
            "Scotty_2-30.h5",
            "Scotty_2-31.h5",
            "Scotty_2-32.h5",
        ],
    ),
    "12-09" => Dict(
        "NoDye_1" => [
            "NoDye_1-1.h5",
            "NoDye_1-9.h5",
            "NoDye_1-19.h5",
            "NoDye_1-23.h5",
        ],
        "NoDye_2" => [
            "NoDye_2-1.h5",
            "NoDye_2-3.h5",
            "NoDye_2-4.h5",
            "NoDye_2-5.h5",
            "NoDye_2-6.h5",
            "NoDye_2-7.h5",
            "NoDye_2-8.h5",
            "NoDye_2-9.h5",
            "NoDye_2-20.h5",
        ],
    ),
    "12-10" => Dict(
        "NoDye_1" => [
            "NoDye_1-14.h5",
            "NoDye_1-19.h5",
            "NoDye_1-20.h5",
            "NoDye_1-23.h5",
        ],
        "NoDye_2" => [
            "NoDye_2-8.h5",
            "NoDye_2-9.h5",
            "NoDye_2-10.h5",
            "NoDye_2-11.h5",
            "NoDye_2-12.h5",
            "NoDye_2-13.h5",
            "NoDye_2-14.h5",
            "NoDye_2-15.h5",
            "NoDye_2-16.h5",
            "NoDye_2-17.h5",
            "NoDye_2-18.h5",
            "NoDye_2-19.h5",
            "NoDye_2-20.h5",
            "NoDye_2-21.h5",
            "NoDye_2-22.h5",
            "NoDye_2-23.h5",
        ]
    ),
)

