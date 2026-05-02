import matplotlib.colors as mcolors
import numpy as np

##############################################
# Title: Radar Plot Configuration
# Description: Colormaps and plotting settings for radar fields (Z, RHOHV, ZDR, KDP)
# Author: Jacob Widanski
# Date: 12 April 2026
##############################################

## Create custom colormaps for ZDR, CC, and KDP from GRAnalyst palettes / .pal files (AI-assisted)

### RHOHV ###
def rho_hv(rho_vmin=0.0, rho_vmax=1.05):
    """
    Correlation Coefficient (RHOHV) colormap from provided palette.
    
    Parameters:
        rho_vmin, rho_vmax : float
            Display limits for RHOHV. Default 0.0 to 1.05.
    """
    rho_points = [
        (rho_vmin, (255,255,255)),
        (0.45, (0,0,0)),
        (0.60, (10,10,190)),
        (0.75, (120,120,255)),
        (0.80, (95,245,100)),
        (0.85, (135,215,10)),
        (0.90, (255,255,0)),
        (0.95, (255,140,0)),
        (0.97, (225,3,0)),
        (0.99, (139,30,77)),
        (1.00, (255,180,215)),
        (rho_vmax, (164,54,150)),
    ]
    pos = [(v - rho_vmin)/(rho_vmax - rho_vmin) for v, _ in rho_points]
    cols = [tuple(np.array(rgb)/255.0) for _, rgb in rho_points]

    rho_cmap = mcolors.LinearSegmentedColormap.from_list("rhohv_custom", list(zip(pos, cols)))
    rho_cmap.set_over((0.5, 0.2, 0.5)) # values above 1.05

    return rho_cmap, [rho_vmin, rho_vmax]

### Velocity ###
def velocity_ms(vel_vmin_mph=-142, vel_vmax_mph=139.8, step_mph=10.0):
    """
    Base Velocity (BV) colormap in m/s.
    Palette anchors are provided in MPH and converted to m/s.
    """
    def rgb01(rgb):
        return np.array(rgb, dtype=float) / 255.0

    mph_per_ms = 2.23694
    vel_vmin_ms = vel_vmin_mph / mph_per_ms
    vel_vmax_ms = vel_vmax_mph / mph_per_ms
    step_ms = step_mph / mph_per_ms

    # value_in_mph, start_rgb, end_rgb
    anchors_mph = [
        (200,  (45, 0, 0),      (45, 0, 0)),
        (140,  (60, 0, 0),      (60, 0, 0)),
        (120,  (97, 6, 2),      (97, 6, 2)),
        (80,   (254, 137, 80),  (254, 137, 80)),
        (60,   (255, 230, 169), (255, 151, 86)),
        (55,   (255, 157, 206), (255, 221, 176)),
        (40,   (249, 58, 84),   (255, 142, 212)),
        (10,   (105, 0, 0),     (242, 1, 6)),
        (0,    (130, 106, 120), (122, 48, 57)),
        (-10,  (72, 112, 71),   (106, 125, 105)),
        (-40,  (10, 248, 35),   (15, 99, 20)),
        (-50,  (180, 240, 243), (33, 253, 50)),
        (-70,  (55, 226, 229),  (172, 239, 242)),
        (-90,  (25, 1, 142),    (47, 215, 225)),
        (-100, (105, 2, 142),   (32, 1, 141)),
        (-120, (250, 4, 130),   (114, 3, 141)),
        (-140, (255, 20, 180),  (255, 20, 180)),
        (-200, (255, 220, 220), (255, 220, 220)),
    ]

    # Convert anchor values to m/s
    anchors = [(v_mph / mph_per_ms, c0, c1) for v_mph, c0, c1 in anchors_mph]
    anchors = sorted(anchors, key=lambda x: x[0])

    xs, cs = [], []
    for i, (lo, c0, c1) in enumerate(anchors):
        hi = anchors[i + 1][0] if i < len(anchors) - 1 else vel_vmax_ms
        lo_clip = max(lo, vel_vmin_ms)
        hi_clip = min(hi, vel_vmax_ms)
        if hi_clip <= lo_clip:
            continue

        n = max(16, int((hi_clip - lo_clip) * 6))
        vals = np.linspace(lo_clip, hi_clip, n, endpoint=False)

        for v in vals:
            t = (v - lo_clip) / (hi_clip - lo_clip)
            c = rgb01(c0) + t * (rgb01(c1) - rgb01(c0))
            x = (v - vel_vmin_ms) / (vel_vmax_ms - vel_vmin_ms)
            xs.append(x)
            cs.append(tuple(c))

    xs.append(1.0)
    cs.append(tuple(rgb01(anchors[-1][2])))

    vel_cmap = mcolors.LinearSegmentedColormap.from_list(
        "bv_velocity_ms_custom", list(zip(xs, cs)), N=256
    )
    vel_cmap.set_under(tuple(rgb01(anchors[0][1])))
    vel_cmap.set_over(tuple(rgb01(anchors[-1][2])))
    vel_cmap.set_bad(tuple(rgb01((123, 0, 200))))  # RF

    #levels = np.arange(vel_vmin_ms, vel_vmax_ms + step_ms, step_ms)
    levels = np.arange(vel_vmin_ms, vel_vmax_ms + 0.5 * step_ms, step_ms)
    return vel_cmap, [vel_vmin_ms, vel_vmax_ms], levels

### ZDR ###
def zdr(zdr_vmin=-7.9, zdr_vmax=7.9):
    """
    Differential Reflectivity (ZDR) colormap from provided palette.
    
    Parameters:
        zdr_vmin, zdr_vmax : float
            Display limits for ZDR. Default -7.9 to 7.9 dB.
    """
    eps = 1e-6  # tiny offset

    zdr_points = [
        (zdr_vmin,  (0, 0, 0)),
        (-6.0,  (55, 55, 55)),
        (-4.0,  (110, 110, 110)),
        (-2.0,  (165, 165, 165)),

        # split zero
        (0.0 - eps, (220, 220, 220)),   # negative side of 0
        (0.0 + eps, (142, 121, 181)),   # positive side of 0

        (0.25, (10, 10, 155)),
        (1.0,  (68, 248, 212)),
        (1.5,  (90, 221, 98)),
        (2.0,  (255, 255, 100)),
        (3.0,  (220, 10, 5)),
        (4.0,  (175, 0, 0)),
        (5.0,  (240, 120, 180)),
        (6.0,  (255, 255, 255)),
        (7.0,  (255, 255, 255)),
        (zdr_vmax,  (255, 255, 255)),
    ]

    zdr_pos = [(v - zdr_vmin) / (zdr_vmax - zdr_vmin) for v, _ in zdr_points]
    zdr_cols = [tuple(np.array(rgb) / 255.0) for _, rgb in zdr_points]
    zdr_cmap = mcolors.LinearSegmentedColormap.from_list("zdr_custom", list(zip(zdr_pos, zdr_cols)))
    zdr_cmap.set_under(zdr_cols[0])
    zdr_cmap.set_over(zdr_cols[-1])
    return zdr_cmap, [zdr_vmin, zdr_vmax]

### KDP ###
def kdp(kdp_vmin=-2.0, kdp_vmax=7.0):
    # KDP continuous colormap from .pal-style segment definitions
    """
    Specific Differential Phase (KDP) colormap from provided palette.
    
    Parameters:
        kdp_vmin, kdp_vmax : float
            Display limits for KDP. Default -2.0 to 7.0.
    """

    def rgb01(rgb):
        return np.array(rgb, dtype=float) / 255.0

    # (lo, hi, start_color, end_color)
    segments = [
        (kdp_vmin, -1.0, (118,118,118), (118,118,118)),  # SolidColor @ kdp_vmin
        (-1.0, -0.5, (75,75,75),    (75,75,75)),     # SolidColor @ -1.0
        (-0.5,  0.0, (75,0,0),      (75,0,0)),       # SolidColor @ -0.5
        ( 0.0,  1.0, (115,0,25),    (213,71,92)),    # Color @ 0.0
        ( 1.0,  1.5, (235,120,185), (155,80,122)),   # Color @ 1.0
        ( 1.5,  2.0, (150,129,183), (100,86,121)),   # Color @ 1.5
        ( 2.0,  2.5, (98,255,250),  (65,170,168)),   # Color @ 2.0
        ( 2.5,  3.0, (20,185,50),   (20,185,50)),    # SolidColor @ 2.5
        ( 3.0,  4.0, (10,255,10),   (10,255,10)),    # SolidColor @ 3.0
        ( 4.0,  5.0, (255,255,0),   (164,164,0)),    # Color @ 4.0
        ( 5.0,  kdp_vmax, (255,120,20),  (164,72,10)),    # Color @ 5.0
    ]

    xs, cs = [], []
    for lo, hi, c0, c1 in segments:
        n = max(16, int((hi - lo) * 40))  # denser sampling = smoother continuous ramp
        vals = np.linspace(lo, hi, n, endpoint=False)
        for v in vals:
            t = (v - lo) / (hi - lo) if hi > lo else 0.0
            c = rgb01(c0) + t * (rgb01(c1) - rgb01(c0))
            xs.append((v - kdp_vmin) / (kdp_vmax - kdp_vmin))
            cs.append(tuple(c))

    # include exact top endpoint color
    xs.append(1.0)
    cs.append(tuple(rgb01((255,205,130))))  # SolidColor @ kdp_vmax

    kdp_cmap = mcolors.LinearSegmentedColormap.from_list("kdp_custom", list(zip(xs, cs)), N=256)
    kdp_cmap.set_under(tuple(rgb01((118,118,118))))
    kdp_cmap.set_over(tuple(rgb01((255,205,130))))

    return kdp_cmap, [kdp_vmin, kdp_vmax]

def get_dict():
    v_cmap, v_lim, v_levels = velocity_ms()

    # Dictionary storing plotting info for each field
    return {
        'z': {
            'vmin': -10,
            'vmax': 70,
            'cmap': "ChaseSpectral",
            'cbar_label': r'$Z_H$ (dBZ)',
            'title': r'Reflectivity ($Z_H$)',
            'units': 'dBZ'
        },
        'v': {
            'vmin': v_lim[0],
            'vmax': v_lim[1],
            'cmap': v_cmap,
            'levels': v_levels,
            'cbar_label': r'$V$ (m s$^{-1}$)',
            'title': r'Base Velocity ($V$)',
            'units': r'm s$^{-1}$'
        },
        'rho': {
            'vmin': rho_hv()[1][0],
            'vmax': rho_hv()[1][1],
            'cmap': rho_hv()[0],
            'cbar_label': r'$\rho_{HV}$',
            'title': r'Correlation Coefficient ($\rho_{HV}$)',
            'units': ''
        },
        'zdr': {
            'vmin': zdr()[1][0],
            'vmax': zdr()[1][1],
            'cmap': zdr()[0],
            'cbar_label': r'$Z_{DR}$ (dB)',
            'title': r'Differential Reflectivity ($Z_{DR}$)',
            'units': 'dB'
        },
        'kdp': {
            'vmin': kdp()[1][0],
            'vmax': kdp()[1][1],
            'cmap': kdp()[0],
            'cbar_label': r'$K_{DP}$ (°/km)',
            'title': r'Specific Differential Phase ($K_{DP}$)',
            'units': r'°/km'
        },
        'hdr_size': {
            'vmin': 0,
            'vmax': 60,
            'cmap': "inferno",
            'cbar_label': r'HDR-derived size (mm)',
            'title': r'Hail Differential Reflectivity (HDR) Size',
            'units': 'mm'
        },
         'hsda': {
            'vmin': -0.5,
            'vmax': 3.5,
            'cmap': mcolors.ListedColormap(['white', 'gold', 'red', 'black']),
            'cbar_label': r'HSDA classes',
            'cbar_ticks': [0, 1, 2, 3],
            'cbar_tick_labels': ['No Hail', 'Small Hail\n(<25 mm)', 'Large Hail\n(25-50 mm)', 'Giant Hail\n(>50 mm)'],
            'title': r'Hail Size Discrimination Algorithm (HSDA)',
            'units': ''
        },
        'mesh_75_mh19': {
            'vmin': 0,
            'vmax': 60,
            'cmap': "inferno",
            'cbar_label': r'MESH (mm)',
            'title': r'Maximum Estimated Size of Hail (MESH) - MH19',
            'units': 'mm'
        },
        'hca': {
            'vmin': -0.5,
            'vmax': 9.5,
            'cmap': "LangRainbow12",
            #'cmap': mcolors.ListedColormap(['grey', 'cyan', 'blue', 'green', 'yellow', 'orange', 'red', 'magenta', 'purple', 'black']),
            'cbar_label': r'HCA classes',
            'cbar_ticks': np.arange(10),
            'cbar_tick_labels': ["NC", "AG", "CR", "LR", "RP", "RN", "VI", "WS", "MH", "IH/HDG"],
            'title': r'Hydrometeor Classification Algorithm (Besic et al. 2016)',
            'units': ''
        }
    }

if __name__ == "__main__":
    # Dictionary storing plotting info for each field
    fields_dict = get_dict()