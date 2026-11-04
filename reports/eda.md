# Exploratory Data Analysis Report: FLOODCAST Real Dataset (v2)

## 1. Executive Summary & Scale Dominance
- **Dataset Version**: `v2`
- **Total River Gauges**: 14 stations across CONUS hydrological regimes.
- **Date Span**: 2000-01-01 to 2023-12-31 (8,766 days per station).
- **Scale Dominance**: **Mississippi River at St. Louis, MO (`07010000`)** dominates raw discharge scale with:
  - Mean discharge: 228,468.2 cfs
  - 90th percentile: 436,500.0 cfs
  - 99th percentile: 682,350.0 cfs
  - Historical peak: 931,000.0 cfs
  - Drainage area: 697,000 sq mi (10,561x larger than smallest basin).

## 2. Per-Station Hydrological & Coverage Statistics
| Station ID | River / Name | Region | Area (sq mi) | Coverage | Missing Q | Mean (cfs) | Median (cfs) | P90 (cfs) | P99 (cfs) | Max (cfs) |
|---|---|---|---|---|---|---|---|---|---|---|
| `01092000` | MERRIMACK R NR GOFFS FALLS, BE | New England | 3,092.0 | 100.0% | 0.0% | 6,027.8 | 4,210.0 | 13,200.0 | 25,835.0 | 72,300.0 |
| `01184000` | CONNECTICUT RIVER AT THOMPSONV | New England | 9,660.0 | 100.0% | 0.0% | 19,010.2 | 14,150.0 | 39,500.0 | 78,105.0 | 122,000.0 |
| `01357500` | MOHAWK RIVER AT COHOES NY | Northeast | 3,450.0 | 100.0% | 0.0% | 6,978.4 | 4,670.0 | 15,250.0 | 36,470.0 | 96,900.0 |
| `01438500` | Delaware River at Montague NJ | Northeast | 3,480.0 | 100.0% | 0.0% | 6,616.5 | 4,550.0 | 13,300.0 | 33,535.0 | 176,000.0 |
| `01570500` | Susquehanna River at Harrisbur | Northeast | 24,100.0 | 100.0% | 0.0% | 37,487.9 | 26,200.0 | 79,800.0 | 185,350.0 | 530,000.0 |
| `01644000` | GOOSE CREEK NEAR LEESBURG, VA | Mid-Atlantic | 332.0 | 100.0% | 0.0% | 336.6 | 174.5 | 724.0 | 2,763.5 | 20,800.0 |
| `01646500` | POTOMAC RIVER NEAR WASH, DC LI | Mid-Atlantic | 11,560.0 | 100.0% | 0.0% | 11,547.7 | 6,955.0 | 25,400.0 | 74,540.0 | 192,000.0 |
| `02085000` | ENO RIVER AT HILLSBOROUGH, NC | Southeast | 66.0 | 100.0% | 0.0% | 53.9 | 17.6 | 112.0 | 653.0 | 3,270.0 |
| `02087500` | NEUSE RIVER NEAR CLAYTON, NC | Southeast | 1,150.0 | 100.0% | 0.0% | 1,036.4 | 504.0 | 2,755.0 | 5,223.5 | 18,800.0 |
| `07010000` | Mississippi River at St. Louis | Midwest | 697,000.0 | 100.0% | 0.0% | 228,468.2 | 184,000.0 | 436,500.0 | 682,350.0 | 931,000.0 |
| `08158000` | Colorado Rv at Austin, TX | Southwest | 39,009.0 | 100.0% | 0.0% | 1,150.1 | 463.0 | 2,070.0 | 14,880.0 | 27,500.0 |
| `08378500` | PECOS RIVER NEAR PECOS, NM | Southwest | 189.0 | 100.0% | 0.0% | 83.8 | 41.0 | 199.5 | 550.4 | 1,260.0 |
| `11264500` | MERCED R A HAPPY ISLES BRIDGE  | Sierra | 181.0 | 100.0% | 0.0% | 360.9 | 105.0 | 1,030.0 | 3,193.5 | 5,660.0 |
| `12119000` | CEDAR RIVER AT RENTON, WA | Pacific NW | 184.0 | 100.0% | 0.0% | 702.3 | 499.0 | 1,510.0 | 2,730.0 | 8,450.0 |

## 3. Physical Ranges & Unit Sanity Checks
- **Precipitation (`precipitation_mm`)**: [0.00, 283.40] mm (Passes sanity [0, 600 mm]: **True**)
- **Maximum Temperature (`temp_max_c`)**: [-18.2, 42.8] °C (Passes sanity [-50, 60 °C]: **True**)
- **Minimum Temperature (`temp_min_c`)**: [-30.0, 30.0] °C (Passes sanity [-50, 60 °C]: **True**)
- **Soil Moisture (`soil_moisture`)**: [0.022, 0.514] m³/m³ (Passes sanity [0, 1 m³/m³]: **True**)

## 4. Static Catchment Attribute Values (v2)
| Station ID | Drainage Area (sq mi) | Mean Elev (ft) | Slope (%) | Elev Range (ft) | Elev Std (ft) | Gauge Alt (ft) |
|---|---|---|---|---|---|---|
| `01646500` | 11,560.0 | 1,368.2 | 0.84% | 3,645.0 | 857.6 | 37.04 |
| `07010000` | 697,000.0 | 2,791.2 | 0.18% | 10,584.0 | 2,329.5 | 379.58 |
| `08158000` | 39,009.0 | 2,286.2 | 0.18% | 3,467.8 | 733.9 | 391.96 |
| `02087500` | 1,150.0 | 450.2 | 0.34% | 531.5 | 147.0 | 127.5 |
| `01092000` | 3,092.0 | 1,030.2 | 1.52% | 3,097.1 | 639.1 | 108.58 |
| `01184000` | 9,660.0 | 1,199.4 | 0.83% | 2,549.2 | 493.7 | 27.75 |
| `01357500` | 3,450.0 | 1,304.0 | 0.92% | 2,326.1 | 575.4 | 48.47 |
| `01438500` | 3,480.0 | 1,649.1 | 1.07% | 2,834.6 | 470.2 | 369.2 |
| `01570500` | 24,100.0 | 1,421.1 | 0.54% | 1,847.1 | 445.4 | 289.34 |
| `01644000` | 332.0 | 631.2 | 1.57% | 1,466.5 | 284.1 | 247.96 |
| `02085000` | 66.0 | 686.3 | 1.03% | 275.6 | 48.7 | 486.7 |
| `08378500` | 189.0 | 10,136.3 | 7.30% | 4,947.5 | 992.7 | 7502.94 |
| `11264500` | 181.0 | 9,018.8 | 8.89% | 5,465.9 | 1,307.3 | 4020.46 |
| `12119000` | 184.0 | 2,014.8 | 6.92% | 4,140.4 | 1,206.3 | 18.77 |

## 5. Static Catchment Attribute Missingness
- `drainage_area_sqmi`: 0.0% missing (USGS NWIS primary, NLDI polygon cross-check).
- `mean_elevation_ft`: 0.0% missing (Open-Meteo elevation API regular grid sampling).
- `slope_pct`: 0.0% missing (adjacent-pair finite difference across regular grid).
- `elev_range_ft`: 0.0% missing (elevation max - min in sampled polygon).
- `elev_std_ft`: 0.0% missing (elevation standard deviation in sampled polygon).
- `forest_cover_pct`: 100.0% missing (no clean public tabular REST API available).
