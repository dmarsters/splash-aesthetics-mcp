"""
Splash Aesthetics MCP Server

Three-layer architecture for zero-cost splash composition:
- Layer 1: Pure taxonomy (splash types, phases, parameters)
- Layer 2: Deterministic mapping (intent → parameters)
- Layer 3: Claude synthesis interface

Phase 2.6: Rhythmic presets with 5 curated oscillation patterns
    - energy_surge (14): drip ↔ explosion (unique gap-filler 12-15)
    - viscosity_sweep (22): water ↔ paint (shared with catastrophe/heraldic)
    - freeze_thaw (18): flow ↔ frozen crown (shared with nuclear/catastrophe/diatom)
    - scatter_convergence (26): focused crown ↔ chaotic spray (unique gap-filler 25-30)
    - chromatic_burst (30): muted drip ↔ vivid paint (LCM hub)

Phase 2.7: Attractor visualization prompt generation
    - 7 discovered/curated attractor presets from Tier 4D analysis
    - Composite, split-view, and sequence prompt modes
    - Domain registry helper for emergent attractor discovery

Based on physics-based fluid dynamics and visual aesthetics.
"""

from fastmcp import FastMCP
import yaml
from pathlib import Path
import json
import math
import re
from typing import Dict, List, Any, Optional

# Initialize FastMCP server
mcp = FastMCP("Splash Aesthetics")

# Load taxonomy files
TAXONOMY_DIR = Path(__file__).parent / "taxonomy"

def load_yaml(filename: str) -> Dict:
    """Load YAML taxonomy file"""
    with open(TAXONOMY_DIR / filename, 'r') as f:
        return yaml.safe_load(f)

# Load all taxonomies at startup
SPLASH_TYPES = load_yaml("splash_types.yaml")
TEMPORAL_PHASES = load_yaml("temporal_phases.yaml")
COMPOSITIONAL_PARAMS = load_yaml("compositional_parameters.yaml")


# ============================================================================
# Phase 2.6/2.7 Parameter Space Definition
# ============================================================================
#
# Normalized 5D morphospace capturing the full range of splash aesthetics.
# Each axis maps to the physics-based compositional dimensions from the
# YAML taxonomy but expressed as continuous [0, 1] coordinates suitable
# for trajectory integration, rhythmic oscillation, and attractor analysis.

SPLASH_PARAMETER_NAMES = [
    "impact_energy",        # 0 = gentle gravity drip → 1 = explosive high-velocity burst
    "fluid_viscosity",      # 0 = water-thin transparent → 1 = thick opaque paint
    "temporal_freeze",      # 0 = long-exposure flow → 1 = frozen-instant sharp edges
    "dispersion_density",   # 0 = sparse minimal droplets → 1 = chaotic saturated scatter
    "chromatic_intensity",  # 0 = monochrome tonal → 1 = high-contrast complementary
]

SPLASH_PARAMETER_BOUNDS = [0.0, 1.0]
SPLASH_DIMENSIONALITY = 5

# ============================================================================
# 7 Canonical Splash States (one per splash type)
# ============================================================================

SPLASH_CANONICAL_STATES: Dict[str, Dict[str, float]] = {
    "crown_splash": {
        "impact_energy": 0.55,
        "fluid_viscosity": 0.10,
        "temporal_freeze": 0.95,
        "dispersion_density": 0.45,
        "chromatic_intensity": 0.80,
    },
    "sheet_splash": {
        "impact_energy": 0.30,
        "fluid_viscosity": 0.05,
        "temporal_freeze": 0.90,
        "dispersion_density": 0.20,
        "chromatic_intensity": 0.50,
    },
    "jet_splash": {
        "impact_energy": 0.80,
        "fluid_viscosity": 0.15,
        "temporal_freeze": 0.95,
        "dispersion_density": 0.50,
        "chromatic_intensity": 0.75,
    },
    "thrown_paint": {
        "impact_energy": 0.65,
        "fluid_viscosity": 0.85,
        "temporal_freeze": 0.40,
        "dispersion_density": 0.70,
        "chromatic_intensity": 0.95,
    },
    "arterial_spray": {
        "impact_energy": 0.85,
        "fluid_viscosity": 0.45,
        "temporal_freeze": 0.85,
        "dispersion_density": 0.90,
        "chromatic_intensity": 0.55,
    },
    "cascade_drip": {
        "impact_energy": 0.15,
        "fluid_viscosity": 0.90,
        "temporal_freeze": 0.20,
        "dispersion_density": 0.15,
        "chromatic_intensity": 0.30,
    },
    "explosive_burst": {
        "impact_energy": 1.00,
        "fluid_viscosity": 0.10,
        "temporal_freeze": 0.95,
        "dispersion_density": 1.00,
        "chromatic_intensity": 0.85,
    },
}

# ============================================================================
# Visual Type Mapping (nearest-neighbor for vocabulary extraction)
# ============================================================================

SPLASH_VISUAL_TYPES = {
    "frozen_crown": {
        "center": {
            "impact_energy": 0.55, "fluid_viscosity": 0.10,
            "temporal_freeze": 0.95, "dispersion_density": 0.45,
            "chromatic_intensity": 0.80,
        },
        "keywords": [
            "razor-sharp corona rim with satellite droplets",
            "radially symmetric liquid crown frozen at peak formation",
            "ballistic droplet trajectories ejected from crown tips",
            "surface tension meniscus catching spectral highlights",
            "high-speed capture of capillary wave interference",
        ],
        "optical": {"finish": "specular_liquid", "scatter": "caustic_refraction", "transparency": "translucent"},
    },
    "gestural_paint": {
        "center": {
            "impact_energy": 0.65, "fluid_viscosity": 0.85,
            "temporal_freeze": 0.40, "dispersion_density": 0.70,
            "chromatic_intensity": 0.95,
        },
        "keywords": [
            "viscous paint ropes whipping through mid-flight arc",
            "gestural thrown pigment with directional momentum streaks",
            "thick opaque strands forming ligament bridges between masses",
            "Nick Knight editorial paint explosion saturated color",
            "abstract expressionist fluid gesture at full chromatic intensity",
        ],
        "optical": {"finish": "glossy_wet_pigment", "scatter": "body_absorption", "transparency": "opaque"},
    },
    "pressurized_jet": {
        "center": {
            "impact_energy": 0.82, "fluid_viscosity": 0.30,
            "temporal_freeze": 0.90, "dispersion_density": 0.70,
            "chromatic_intensity": 0.65,
        },
        "keywords": [
            "pressurized directional stream breaking into spray cone",
            "high-velocity jet impact with radial splatter pattern",
            "atomized mist halo surrounding coherent liquid core",
            "ballistic fragmentation cascade from nozzle exit",
            "forensic-precision directional spray field frozen in flight",
        ],
        "optical": {"finish": "wet_sheen", "scatter": "mist_diffusion", "transparency": "mixed"},
    },
    "viscous_flow": {
        "center": {
            "impact_energy": 0.15, "fluid_viscosity": 0.90,
            "temporal_freeze": 0.20, "dispersion_density": 0.15,
            "chromatic_intensity": 0.30,
        },
        "keywords": [
            "gravity-drawn viscous thread descending in slow catenary",
            "honey-like pooling with concentric fold patterns",
            "long-exposure silk-smooth fluid surface with motion blur",
            "contemplative single drip moment before surface contact",
            "tonal monochrome thick medium catching soft directional light",
        ],
        "optical": {"finish": "satin_viscous", "scatter": "subsurface_glow", "transparency": "semi_translucent"},
    },
    "explosive_scatter": {
        "center": {
            "impact_energy": 0.95, "fluid_viscosity": 0.10,
            "temporal_freeze": 0.95, "dispersion_density": 0.95,
            "chromatic_intensity": 0.85,
        },
        "keywords": [
            "omnidirectional liquid shrapnel frozen at maximum dispersion",
            "chaotic droplet field with no coherent structure or axis",
            "micro-droplet constellation suspended against black void",
            "explosive fragmentation with size gradient from core to edge",
            "maximum entropy fluid state captured at 1/10000 shutter speed",
        ],
        "optical": {"finish": "specular_micro", "scatter": "individual_droplet", "transparency": "translucent"},
    },
}

# ============================================================================
# Visual Vocabulary (parameter-indexed descriptors for prompt generation)
# ============================================================================

SPLASH_VISUAL_VOCABULARY = {
    "fluid_dynamics": [
        "gravity-drawn slow descent in catenary curve",
        "gentle laminar flow with smooth unbroken surface",
        "moderate impact splash with controlled spread",
        "energetic collision generating radial wave fronts",
        "violent rupture with ballistic fragment ejection",
        "high-pressure jet breaking into atomized spray cone",
        "explosive omnidirectional burst at terminal velocity",
        "supercritical shattering into micro-droplet constellation",
    ],
    "surface_interaction": [
        "pooling with concentric ripple rings",
        "thin film spreading across flat surface",
        "crown formation with satellite droplet ejection",
        "sheet breakup into ligament fingers",
        "spray rebound with secondary droplet generation",
        "cavitation void collapse sending vertical jet",
        "crater formation with ejecta curtain",
        "surface obliteration with no coherent pool remaining",
    ],
    "fragmentation": [
        "single intact body with smooth unbroken meniscus",
        "minor thread formation at trailing edge",
        "ligament bridges connecting main masses",
        "droplet pinch-off from thinning filaments",
        "spray generation from rim instability",
        "fine mist halo surrounding coherent core",
        "complete atomization into polydisperse cloud",
        "shattered liquid confetti at maximum entropy",
    ],
    "light_capture": [
        "matte diffuse absorption on opaque thick medium",
        "satin sheen on viscous surface with soft highlights",
        "wet gloss with environmental reflections on pooled surface",
        "spectral caustics refracting through translucent body",
        "individual droplet lensing with bokeh backlight",
        "flash-frozen specular highlights on every micro-surface",
        "total internal reflection in thin stretched membranes",
        "chromatic dispersion through airborne prism droplets",
    ],
    "temporal_quality": [
        "long-exposure silk-smooth motion blur dissolving edges",
        "moderate blur preserving directional momentum streaks",
        "slight motion trace at extremity tips only",
        "crisp edges with minimal motion artifact",
        "razor-sharp freeze at 1/4000 capturing every ligament",
        "micro-flash frozen at 1/10000 suspending all motion",
        "stroboscopic multi-exposure showing trajectory path",
        "bullet-time frozen with perfect volumetric sharpness",
    ],
    "color_behavior": [
        "achromatic tonal grayscale with luminance contrast only",
        "desaturated muted palette with restrained color warmth",
        "natural fluid color — clear water or amber resin",
        "moderate saturation with distinct hue identity",
        "rich pigmented color maintaining purity through thinning",
        "high-contrast complementary pair in collision or layering",
        "maximum chromatic intensity — pure saturated pigment",
        "iridescent color-shifting through thin-film interference",
    ],
}

# ============================================================================
# Phase 2.6 Rhythmic Presets
# ============================================================================
#
# Periods chosen to create productive interactions with other domains:
#   14: Unique — fills gap 12-15 in period landscape
#   18: Shared with nuclear, catastrophe, diatom
#   22: Shared with catastrophe, heraldic
#   26: Unique — fills gap 25-30
#   30: Shared with microscopy, diatom, heraldic, surface_design (LCM hub)

SPLASH_RHYTHMIC_PRESETS = {
    "energy_surge": {
        "period": 14,
        "state_a": "cascade_drip",
        "state_b": "explosive_burst",
        "pattern": "triangular",
        "description": (
            "Gentle gravity drip ramping to maximum explosive energy — "
            "full dynamic range of fluid impact. Unique period 14 fills "
            "gap between 12 and 15 for novel beat frequencies."
        ),
    },
    "viscosity_sweep": {
        "period": 22,
        "state_a": "sheet_splash",
        "state_b": "thrown_paint",
        "pattern": "sinusoidal",
        "description": (
            "Water-thin transparent sheet dissolving into thick opaque paint — "
            "material character transformation from delicate to gestural. "
            "Syncs with catastrophe/heraldic period 22."
        ),
    },
    "freeze_thaw": {
        "period": 18,
        "state_a": "cascade_drip",
        "state_b": "crown_splash",
        "pattern": "sinusoidal",
        "description": (
            "Slow viscous flow melting into sharp frozen crown — "
            "temporal capture oscillation from long-exposure blur to "
            "razor-sharp instant. Syncs with nuclear/catastrophe/diatom period 18."
        ),
    },
    "scatter_convergence": {
        "period": 26,
        "state_a": "crown_splash",
        "state_b": "arterial_spray",
        "pattern": "sinusoidal",
        "description": (
            "Ordered radial crown structure dispersing into chaotic spray field — "
            "symmetry to entropy transition. Unique period 26 fills gap 25-30 "
            "for novel compositional emergence."
        ),
    },
    "chromatic_burst": {
        "period": 30,
        "state_a": "cascade_drip",
        "state_b": "thrown_paint",
        "pattern": "sinusoidal",
        "description": (
            "Muted monochrome drip blooming into fully saturated gestural "
            "paint explosion — from contemplative to maximum chromatic impact. "
            "MAJOR LCM HUB for full-system synchronization with period 30 domains."
        ),
    },
}

# ============================================================================
# Phase 2.7 Attractor Presets (Tier 4D Discoveries)
# ============================================================================
#
# Multi-domain emergent attractors mapped to splash parameter coordinates.
# Each preset encodes the splash aesthetic state that arises when the
# system locks into that attractor period during multi-domain composition.
#
# Classification:
#   lcm_sync  — LCM synchronization across 3+ domains
#   novel     — Emergent period not explainable by LCM or harmonics
#   harmonic  — Integer multiple of individual domain periods
#   curated   — Hand-selected edge states for specific aesthetic effects

SPLASH_ATTRACTOR_PRESETS = {
    # ── Tier 1: Stable Cores ──────────────────────────────────────────
    "period_30": {
        "name": "Period 30 — Universal Sync",
        "description": (
            "Dominant LCM synchronization. Splash chromatic_burst preset "
            "(period 30) locks directly into this attractor. Surface sits "
            "in the gestural-paint territory — thick opaque strands with "
            "high color intensity, the most stable multi-domain splash state."
        ),
        "basin_size": 0.116,
        "classification": "lcm_sync",
        "source_domains": ["microscopy", "diatom", "heraldic", "surface_design", "splash"],
        "state": {
            "impact_energy": 0.45,
            "fluid_viscosity": 0.55,
            "temporal_freeze": 0.50,
            "dispersion_density": 0.42,
            "chromatic_intensity": 0.68,
        },
    },
    "period_29": {
        "name": "Period 29 — Emergent Resonance",
        "description": (
            "Purely emergent 5-domain attractor. Splash character is a "
            "mid-energy translucent jet — between crown and sheet, with "
            "moderate freeze and natural fluid color. This aesthetic exists "
            "nowhere in any single splash type."
        ),
        "basin_size": 0.084,
        "classification": "lcm_sync",
        "source_domains": ["microscopy", "nuclear", "catastrophe", "diatom", "heraldic"],
        "state": {
            "impact_energy": 0.50,
            "fluid_viscosity": 0.30,
            "temporal_freeze": 0.65,
            "dispersion_density": 0.35,
            "chromatic_intensity": 0.55,
        },
    },
    "period_19": {
        "name": "Period 19 — Gap Flow",
        "description": (
            "Resilient novel gap-filler between periods 18 and 20. "
            "Splash is a frozen medium-energy impact with controlled "
            "dispersion — the clean analytical splash of scientific "
            "high-speed photography. Prime-period irrational beats."
        ),
        "basin_size": 0.074,
        "classification": "novel",
        "source_domains": ["microscopy", "nuclear", "catastrophe", "diatom"],
        "state": {
            "impact_energy": 0.60,
            "fluid_viscosity": 0.20,
            "temporal_freeze": 0.85,
            "dispersion_density": 0.50,
            "chromatic_intensity": 0.60,
        },
    },
    # ── Tier 2: Specialized ───────────────────────────────────────────
    "period_28": {
        "name": "Period 28 — Composite Beat",
        "description": (
            "Novel composite beat (Period 60 − 2×16 = 28). Splash sits "
            "in tension between frozen precision and gestural motion — "
            "a jet caught in the instant between coherent stream and "
            "spray breakup. The fluid-dynamics equivalent of a held breath."
        ),
        "basin_size": 0.024,
        "classification": "novel",
        "source_domains": ["microscopy", "nuclear", "catastrophe", "diatom"],
        "state": {
            "impact_energy": 0.70,
            "fluid_viscosity": 0.35,
            "temporal_freeze": 0.75,
            "dispersion_density": 0.55,
            "chromatic_intensity": 0.65,
        },
    },
    "period_60": {
        "name": "Period 60 — Harmonic Hub",
        "description": (
            "Major LCM hub (3×20, 4×15, 5×12). Splash oscillates through "
            "full fluid repertoire — every canonical splash state gets a "
            "moment in the long cycle. Complex synchronization, advanced use."
        ),
        "basin_size": 0.040,
        "classification": "harmonic",
        "source_domains": ["microscopy", "nuclear", "catastrophe", "diatom"],
        "state": {
            "impact_energy": 0.55,
            "fluid_viscosity": 0.40,
            "temporal_freeze": 0.60,
            "dispersion_density": 0.48,
            "chromatic_intensity": 0.58,
        },
    },
    # ── Tier 3: Curated Edge States ───────────────────────────────────
    "bifurcation_edge": {
        "name": "Bifurcation Edge — Breakup Threshold",
        "description": (
            "Curated state at the exact moment a coherent jet breaks into "
            "spray. Poised between pressurized_jet and explosive_scatter "
            "basins — the Plateau-Rayleigh instability threshold where "
            "surface tension loses to inertia."
        ),
        "basin_size": None,
        "classification": "curated",
        "source_domains": ["splash"],
        "state": {
            "impact_energy": 0.78,
            "fluid_viscosity": 0.18,
            "temporal_freeze": 0.92,
            "dispersion_density": 0.68,
            "chromatic_intensity": 0.70,
        },
    },
    "organic_complexity": {
        "name": "Organic Complexity — Living Fluid",
        "description": (
            "Curated state at maximum organic complexity. Thick botanical "
            "fluid with subsurface color — honey, nectar, tree resin — "
            "caught in slow gravitational descent with natural warmth "
            "and translucent depth."
        ),
        "basin_size": None,
        "classification": "curated",
        "source_domains": ["splash"],
        "state": {
            "impact_energy": 0.25,
            "fluid_viscosity": 0.80,
            "temporal_freeze": 0.30,
            "dispersion_density": 0.22,
            "chromatic_intensity": 0.45,
        },
    },
}


# ============================================================================
# Phase 2.6/2.7 Helper Functions
# ============================================================================

def _splash_euclidean_distance(a: Dict[str, float], b: Dict[str, float]) -> float:
    """Euclidean distance between two states in splash parameter space."""
    return math.sqrt(sum((a[p] - b[p]) ** 2 for p in SPLASH_PARAMETER_NAMES))


def _splash_nearest_canonical(state: Dict[str, float]) -> tuple:
    """Find nearest canonical splash state. Returns (state_id, distance)."""
    best_id, best_dist = None, float("inf")
    for sid, coords in SPLASH_CANONICAL_STATES.items():
        d = _splash_euclidean_distance(state, coords)
        if d < best_dist:
            best_id, best_dist = sid, d
    return best_id, best_dist


def _splash_nearest_visual_type(state: Dict[str, float]) -> tuple:
    """Find nearest splash visual type. Returns (type_id, distance)."""
    best_id, best_dist = None, float("inf")
    for vid, vdata in SPLASH_VISUAL_TYPES.items():
        d = _splash_euclidean_distance(state, vdata["center"])
        if d < best_dist:
            best_id, best_dist = vid, d
    return best_id, best_dist


def _splash_interpolate_states(
    state_a: Dict[str, float],
    state_b: Dict[str, float],
    t: float,
    pattern: str = "sinusoidal",
) -> Dict[str, float]:
    """Interpolate between two splash states at phase t ∈ [0, 1]."""
    if pattern == "sinusoidal":
        alpha = 0.5 * (1.0 - math.cos(math.pi * t))
    elif pattern == "triangular":
        alpha = 2.0 * t if t <= 0.5 else 2.0 * (1.0 - t)
    elif pattern == "square":
        alpha = 0.0 if t < 0.5 else 1.0
    else:
        alpha = t
    return {
        p: state_a[p] + alpha * (state_b[p] - state_a[p])
        for p in SPLASH_PARAMETER_NAMES
    }


def _splash_select_vocabulary(state: Dict[str, float]) -> Dict[str, List[str]]:
    """Select vocabulary terms weighted by splash parameter values."""
    selected = {}
    for category, terms in SPLASH_VISUAL_VOCABULARY.items():
        # Map primary parameter to index
        if category == "fluid_dynamics":
            idx = round(state["impact_energy"] * (len(terms) - 1))
        elif category == "surface_interaction":
            idx = round(state["impact_energy"] * (len(terms) - 1))
        elif category == "fragmentation":
            idx = round(state["dispersion_density"] * (len(terms) - 1))
        elif category == "light_capture":
            # Combine specularity-like axes: viscosity inverted + freeze
            v = (1.0 - state["fluid_viscosity"] + state["temporal_freeze"]) / 2.0
            idx = round(v * (len(terms) - 1))
        elif category == "temporal_quality":
            idx = round(state["temporal_freeze"] * (len(terms) - 1))
        elif category == "color_behavior":
            idx = round(state["chromatic_intensity"] * (len(terms) - 1))
        else:
            idx = 0

        idx = max(0, min(idx, len(terms) - 1))
        neighbors = [
            terms[max(0, idx - 1)],
            terms[idx],
            terms[min(len(terms) - 1, idx + 1)],
        ]
        selected[category] = list(dict.fromkeys(neighbors))  # deduplicate
    return selected


# ============================================================================
# Phase 2.8: Aesthetic Decomposition Helpers
# ============================================================================
#
# Inverse of the generative pipeline: text description → domain coordinates.
# Layer 2: deterministic, 0 LLM tokens.
# Uses keyword matching against SPLASH_VISUAL_TYPES to recover coordinates.

_SPLASH_STOP_WORDS = frozenset({
    'a', 'an', 'the', 'in', 'on', 'at', 'to', 'of', 'for', 'with',
    'by', 'from', 'and', 'or', 'but', 'as', 'is', 'are', 'was', 'were',
    'be', 'been', 'being', 'has', 'have', 'had', 'do', 'does', 'did',
    'no', 'not', 'all', 'its', 'this', 'that', 'into', 'over',
})


def _splash_extract_fragments(keyword: str) -> List[str]:
    """Extract matchable sub-phrases from a keyword string.

    Sliding window of 2-4 words plus full keyword (if 3+ words).
    Skips fragments that are mostly stop words.
    """
    words = keyword.lower().split()
    fragments = []

    if len(words) >= 3:
        fragments.append(keyword.lower())

    for window_size in [4, 3, 2]:
        for i in range(len(words) - window_size + 1):
            frag = ' '.join(words[i:i + window_size])
            content_words = [w for w in words[i:i + window_size]
                             if len(w) > 3 and w not in _SPLASH_STOP_WORDS]
            if len(content_words) >= 1:
                fragments.append(frag)

    return fragments


def _splash_score_visual_type(
    vtype_data: Dict,
    words: set,
    full_text: str,
    substring_weight: float = 1.0,
    word_overlap_weight: float = 0.3,
    optical_weight: float = 0.5,
) -> tuple:
    """Score a visual type against input text. Returns (score, matched_fragments)."""
    score = 0.0
    matched = []

    # Keyword fragment matching
    for keyword in vtype_data.get("keywords", []):
        fragments = _splash_extract_fragments(keyword)
        best_frag_score = 0.0
        best_frag = None

        for frag in fragments:
            if frag in full_text:
                if substring_weight > best_frag_score:
                    best_frag_score = substring_weight
                    best_frag = frag
            else:
                frag_words = set(frag.split()) - _SPLASH_STOP_WORDS
                if frag_words:
                    overlap = len(frag_words & words) / len(frag_words)
                    word_score = overlap * word_overlap_weight
                    if word_score > best_frag_score:
                        best_frag_score = word_score
                        best_frag = frag

        if best_frag and best_frag_score > 0:
            score += best_frag_score
            matched.append(best_frag)

    # Optical property matching
    for prop_name, prop_value in vtype_data.get("optical", {}).items():
        prop_words = set(prop_value.lower().replace('_', ' ').split())
        prop_overlap = len(prop_words & words)
        if prop_overlap > 0:
            score += optical_weight * (prop_overlap / len(prop_words))
            matched.append(f"optical:{prop_value}")

    return score, matched


def _splash_softmax(scores: Dict[str, float], temperature: float = 1.5) -> Dict[str, float]:
    """Softmax with temperature over type scores."""
    if not scores or max(scores.values()) == 0:
        n = len(scores)
        return {k: 1.0 / n for k in scores} if n > 0 else {}

    max_s = max(scores.values())
    exps = {k: math.exp((v - max_s) / temperature) for k, v in scores.items()}
    total = sum(exps.values())
    return {k: v / total for k, v in exps.items()}


def _splash_blend_coordinates(weights: Dict[str, float]) -> Dict[str, float]:
    """Weighted average of visual type centers."""
    result = {p: 0.0 for p in SPLASH_PARAMETER_NAMES}
    for vid, vdata in SPLASH_VISUAL_TYPES.items():
        w = weights.get(vid, 0)
        if w > 0:
            for p in SPLASH_PARAMETER_NAMES:
                result[p] += w * vdata["center"].get(p, 0)
    return result


# ============================================================================
# LAYER 1: Pure Taxonomy (0 tokens)
# ============================================================================

@mcp.tool()
def list_splash_types() -> str:
    """
    List all splash types with physics-based classification.
    
    LAYER 1: Pure taxonomy enumeration (0 LLM tokens).
    
    Returns overview of 7 splash types:
    - Crown splash (radial symmetry, product photography)
    - Sheet splash (thin spreading, delicate)
    - Jet splash (vertical column, dramatic)
    - Thrown paint (gestural, Nick Knight fashion)
    - Arterial spray (pressurized, forensic)
    - Cascade drip (gravity-driven, viscous)
    - Explosive burst (omnidirectional chaos)
    
    Returns:
        JSON string with all splash type specifications
    """
    result = {
        "splash_types": {},
        "total_types": len(SPLASH_TYPES["splash_types"])
    }
    
    for splash_id, splash_data in SPLASH_TYPES["splash_types"].items():
        result["splash_types"][splash_id] = {
            "name": splash_data["name"],
            "description": splash_data["description"],
            "velocity_range": splash_data["physics"]["velocity_range"],
            "symmetry": splash_data["morphology"]["symmetry"],
            "visual_characteristics": splash_data["visual_characteristics"],
            "use_cases": splash_data["use_cases"]
        }
    
    return json.dumps(result, indent=2)


@mcp.tool()
def get_splash_type_details(splash_type: str) -> str:
    """
    Get complete specification for a specific splash type.
    
    LAYER 1: Pure taxonomy retrieval (0 LLM tokens).
    
    Args:
        splash_type: One of: crown_splash, sheet_splash, jet_splash,
                    thrown_paint, arterial_spray, cascade_drip, explosive_burst
    
    Returns:
        JSON string with complete physics, morphology, and visual specs
    """
    if splash_type not in SPLASH_TYPES["splash_types"]:
        return json.dumps({
            "error": f"Unknown splash type: {splash_type}",
            "available_types": list(SPLASH_TYPES["splash_types"].keys())
        })
    
    return json.dumps(SPLASH_TYPES["splash_types"][splash_type], indent=2)


@mcp.tool()
def list_temporal_phases() -> str:
    """
    List all temporal phases of splash evolution.
    
    LAYER 1: Pure temporal structure (0 LLM tokens).
    
    Returns phases from pre-impact through settling:
    - Pre-impact trajectory
    - Impact moment (peak energy)
    - Crown formation (geometric beauty)
    - Droplet ejection (fragmentation)
    - Secondary splash (cascade)
    - Settling phase (return to equilibrium)
    
    Each phase maps to catastrophe theory transformations.
    
    Returns:
        JSON string with all temporal phase specifications
    """
    result = {
        "temporal_phases": {},
        "sequences": TEMPORAL_PHASES["temporal_sequences"],
        "total_phases": len(TEMPORAL_PHASES["temporal_phases"])
    }
    
    for phase_id, phase_data in TEMPORAL_PHASES["temporal_phases"].items():
        result["temporal_phases"][phase_id] = {
            "name": phase_data["name"],
            "description": phase_data["description"],
            "duration": phase_data["duration_typical"],
            "energy_state": phase_data["energy_state"],
            "catastrophe_mapping": phase_data["catastrophe_mapping"],
            "compositional_focus": phase_data["compositional_focus"]
        }
    
    return json.dumps(result, indent=2)


@mcp.tool()
def list_compositional_parameters() -> str:
    """
    List all compositional parameter categories.
    
    LAYER 1: Pure parameter space enumeration (0 LLM tokens).
    
    Categories:
    - Color contrast (complementary, analogous, monochromatic)
    - Density distribution (sparse → chaotic)
    - Scale hierarchy (macro → microscopic)
    - Directional bias (radial, diagonal, vertical, horizontal)
    - Opacity/translucency (opaque → transparent)
    - Motion capture (frozen → long exposure)
    
    Returns:
        JSON string with all compositional parameters
    """
    result = {
        "parameter_categories": {
            "color_contrast": list(COMPOSITIONAL_PARAMS["color_contrast"].keys()),
            "density_distribution": list(COMPOSITIONAL_PARAMS["density_distribution"].keys()),
            "scale_hierarchy": list(COMPOSITIONAL_PARAMS["scale_hierarchy"].keys()),
            "directional_bias": list(COMPOSITIONAL_PARAMS["directional_bias"].keys()),
            "opacity_translucency": list(COMPOSITIONAL_PARAMS["opacity_translucency"].keys()),
            "motion_capture": list(COMPOSITIONAL_PARAMS["motion_capture"].keys())
        }
    }
    
    return json.dumps(result, indent=2)


@mcp.tool()
def get_compositional_parameter_details(category: str, parameter: str) -> str:
    """
    Get detailed specification for a compositional parameter.
    
    LAYER 1: Pure taxonomy retrieval (0 LLM tokens).
    
    Args:
        category: Parameter category (color_contrast, density_distribution, etc.)
        parameter: Specific parameter within category
    
    Returns:
        JSON string with complete parameter specification
    """
    if category not in COMPOSITIONAL_PARAMS:
        return json.dumps({
            "error": f"Unknown category: {category}",
            "available_categories": list(COMPOSITIONAL_PARAMS.keys())
        })
    
    if parameter not in COMPOSITIONAL_PARAMS[category]:
        return json.dumps({
            "error": f"Unknown parameter: {parameter}",
            "available_parameters": list(COMPOSITIONAL_PARAMS[category].keys())
        })
    
    return json.dumps(COMPOSITIONAL_PARAMS[category][parameter], indent=2)


# ============================================================================
# LAYER 2: Deterministic Mapping (0 tokens)
# ============================================================================

@mcp.tool()
def classify_splash_intent(user_intent: str) -> str:
    """
    Classify splash intent from user description.
    
    LAYER 2: Deterministic keyword matching (0 LLM tokens).
    
    Analyzes intent text and matches against splash type keywords
    to determine the most appropriate splash classification.
    
    Args:
        user_intent: User's description of desired splash aesthetic
    
    Returns:
        JSON string with classification results including:
        - primary_splash_type: Best match splash type
        - confidence: Match confidence (0.0-1.0)
        - matched_keywords: Keywords that triggered classification
        - alternative_types: Other potential matches
    """
    intent_lower = user_intent.lower()
    
    # Score each splash type based on keyword matches
    scores = {}
    keyword_matches = {}
    
    for splash_type, keywords in SPLASH_TYPES["intent_keywords"].items():
        matches = [kw for kw in keywords if kw in intent_lower]
        if matches:
            scores[splash_type] = len(matches)
            keyword_matches[splash_type] = matches
    
    if not scores:
        return json.dumps({
            "primary_splash_type": "crown_splash",  # Default
            "confidence": 0.3,
            "reason": "No clear keywords matched, defaulting to crown splash",
            "matched_keywords": [],
            "alternative_types": list(SPLASH_TYPES["splash_types"].keys())
        })
    
    # Get primary type (highest score)
    primary_type = max(scores.items(), key=lambda x: x[1])[0]
    max_score = scores[primary_type]
    
    # Calculate confidence
    confidence = min(max_score / 5.0, 1.0)  # Normalize by typical max keywords
    
    # Get alternatives
    alternatives = sorted(
        [(t, s) for t, s in scores.items() if t != primary_type],
        key=lambda x: x[1],
        reverse=True
    )[:2]
    
    return json.dumps({
        "primary_splash_type": primary_type,
        "confidence": round(confidence, 2),
        "matched_keywords": keyword_matches[primary_type],
        "alternative_types": [{"type": t, "score": s} for t, s in alternatives],
        "splash_details": {
            "name": SPLASH_TYPES["splash_types"][primary_type]["name"],
            "description": SPLASH_TYPES["splash_types"][primary_type]["description"]
        }
    }, indent=2)


@mcp.tool()
def map_intent_to_parameters(
    user_intent: str,
    splash_type: Optional[str] = None
) -> str:
    """
    Map user intent to complete splash parameters.
    
    LAYER 2: Deterministic parameter mapping (0 LLM tokens).
    
    Analyzes intent and determines appropriate compositional parameters
    across all categories: color, density, scale, direction, opacity, motion.
    
    Args:
        user_intent: User's description of desired aesthetic
        splash_type: Optional specific splash type (auto-detected if not provided)
    
    Returns:
        JSON string with complete parameter mapping
    """
    # Auto-detect splash type if not provided
    if splash_type is None:
        classification = json.loads(classify_splash_intent(user_intent))
        splash_type = classification["primary_splash_type"]
    
    intent_lower = user_intent.lower()
    
    # Map color contrast
    if any(word in intent_lower for word in ["complementary", "contrast", "vibrant", "opposite"]):
        color = "complementary_high"
    elif any(word in intent_lower for word in ["harmony", "analogous", "flowing"]):
        color = "analogous_harmony"
    elif any(word in intent_lower for word in ["monochrome", "single", "tonal"]):
        color = "monochromatic_tonal"
    elif any(word in intent_lower for word in ["black", "white", "bw"]):
        color = "black_white_pure"
    else:
        color = "complementary_high"  # Default high impact
    
    # Map density
    if any(word in intent_lower for word in ["sparse", "minimal", "few", "delicate"]):
        density = "sparse_minimal"
    elif any(word in intent_lower for word in ["chaos", "explosive", "maximum", "violent"]):
        density = "chaotic_explosive"
    elif any(word in intent_lower for word in ["dense", "heavy", "saturated"]):
        density = "dense_saturated"
    else:
        density = "balanced_medium"
    
    # Map scale
    if any(word in intent_lower for word in ["macro", "close", "detail", "intimate"]):
        scale = "macro_intimate"
    elif any(word in intent_lower for word in ["portrait", "fashion", "face"]):
        scale = "medium_portrait"
    elif any(word in intent_lower for word in ["wide", "scene", "environment", "context"]):
        scale = "environmental_wide"
    elif any(word in intent_lower for word in ["microscopic", "abstract", "pattern"]):
        scale = "microscopic_abstract"
    else:
        scale = "medium_portrait"
    
    # Map directional bias
    if any(word in intent_lower for word in ["radial", "symmetric", "centered"]):
        direction = "radial_symmetric"
    elif any(word in intent_lower for word in ["diagonal", "dynamic"]):
        direction = "diagonal_dynamic"
    elif any(word in intent_lower for word in ["up", "vertical", "lift"]):
        direction = "vertical_lift"
    elif any(word in intent_lower for word in ["horizontal", "sweep", "side"]):
        direction = "horizontal_sweep"
    elif any(word in intent_lower for word in ["chaos", "random", "explosive"]):
        direction = "omnidirectional_chaos"
    else:
        direction = "radial_symmetric"
    
    # Map opacity
    if any(word in intent_lower for word in ["paint", "opaque", "solid", "bold"]):
        opacity = "opaque_solid"
    elif any(word in intent_lower for word in ["glow", "translucent", "luminous"]):
        opacity = "translucent_color"
    elif any(word in intent_lower for word in ["clear", "water", "transparent"]):
        opacity = "transparent_clear"
    else:
        opacity = "translucent_color"
    
    # Map motion capture
    if any(word in intent_lower for word in ["freeze", "frozen", "instant", "sharp"]):
        motion = "frozen_instant"
    elif any(word in intent_lower for word in ["blur", "streak", "motion"]):
        motion = "motion_streaks"
    elif any(word in intent_lower for word in ["flow", "long exposure", "ethereal"]):
        motion = "long_exposure_flow"
    else:
        motion = "frozen_instant"  # Default for splash photography
    
    # Get temporal phase emphasis
    phase_emphasis = determine_phase_emphasis(intent_lower, splash_type)
    
    return json.dumps({
        "splash_type": splash_type,
        "splash_details": SPLASH_TYPES["splash_types"][splash_type],
        "compositional_parameters": {
            "color_contrast": {
                "selected": color,
                "details": COMPOSITIONAL_PARAMS["color_contrast"][color]
            },
            "density_distribution": {
                "selected": density,
                "details": COMPOSITIONAL_PARAMS["density_distribution"][density]
            },
            "scale_hierarchy": {
                "selected": scale,
                "details": COMPOSITIONAL_PARAMS["scale_hierarchy"][scale]
            },
            "directional_bias": {
                "selected": direction,
                "details": COMPOSITIONAL_PARAMS["directional_bias"][direction]
            },
            "opacity_translucency": {
                "selected": opacity,
                "details": COMPOSITIONAL_PARAMS["opacity_translucency"][opacity]
            },
            "motion_capture": {
                "selected": motion,
                "details": COMPOSITIONAL_PARAMS["motion_capture"][motion]
            }
        },
        "temporal_emphasis": phase_emphasis
    }, indent=2)


def determine_phase_emphasis(intent_lower: str, splash_type: str) -> Dict[str, Any]:
    """Helper: Determine temporal phase emphasis from intent"""
    
    if any(word in intent_lower for word in ["energy", "dramatic", "impact"]):
        emphasis = "high_energy"
    elif any(word in intent_lower for word in ["beautiful", "geometric", "elegant"]):
        emphasis = "geometric_beauty"
    elif any(word in intent_lower for word in ["calm", "contemplative", "settling"]):
        emphasis = "contemplative"
    elif any(word in intent_lower for word in ["chaos", "explosive", "violent"]):
        emphasis = "chaotic_energy"
    elif any(word in intent_lower for word in ["anticipation", "before", "about to"]):
        emphasis = "anticipation"
    else:
        emphasis = "geometric_beauty"  # Default
    
    phase_data = TEMPORAL_PHASES["phase_emphasis"][emphasis]
    primary_phase = phase_data["primary_phase"]
    
    return {
        "emphasis_mode": emphasis,
        "primary_phase": primary_phase,
        "phase_details": TEMPORAL_PHASES["temporal_phases"][primary_phase],
        "recommended_timing": TEMPORAL_PHASES["temporal_phases"][primary_phase]["duration_typical"]
    }


# ============================================================================
# LAYER 3: Claude Synthesis Interface
# ============================================================================

@mcp.tool()
def enhance_splash_prompt(
    user_intent: str,
    splash_type: Optional[str] = None,
    color_override: Optional[str] = None,
    intensity: str = "moderate"
) -> str:
    """
    Prepare complete splash aesthetic enhancement for Claude synthesis.
    
    LAYER 3 INTERFACE: Combines Layer 1 & 2 outputs into structured
    data ready for Claude to synthesize into an enhanced image prompt.
    
    This tool does NOT synthesize the final prompt - it provides all
    deterministic parameters for Claude to creatively compose.
    
    Args:
        user_intent: User's description of desired splash aesthetic
        splash_type: Optional specific splash type (auto-detected if not provided)
        color_override: Optional specific color scheme
        intensity: Enhancement intensity (subtle, moderate, dramatic)
    
    Returns:
        JSON string with complete enhancement package for Claude synthesis
    """
    # Get complete parameter mapping
    parameters = json.loads(map_intent_to_parameters(user_intent, splash_type))
    
    # Override color if specified
    if color_override and color_override in COMPOSITIONAL_PARAMS["color_contrast"]:
        parameters["compositional_parameters"]["color_contrast"] = {
            "selected": color_override,
            "details": COMPOSITIONAL_PARAMS["color_contrast"][color_override]
        }
    
    # Determine vocabulary intensity
    intensity_multipliers = {
        "subtle": 0.6,
        "moderate": 1.0,
        "dramatic": 1.5
    }
    
    multiplier = intensity_multipliers.get(intensity, 1.0)
    
    result = {
        "original_intent": user_intent,
        "intensity": intensity,
        "splash_specification": parameters,
        "synthesis_instructions": {
            "task": "Synthesize image generation prompt from deterministic parameters",
            "approach": "Translate taxonomy into concrete visual descriptors",
            "emphasis": f"Apply {intensity} intensity to vocabulary selection",
            "guidelines": [
                "Use splash_specification physics for technical accuracy",
                "Translate compositional_parameters into image gen vocabulary",
                "Reference temporal_emphasis for timing/phase capture",
                "Maintain geometric specifications from directional_bias",
                "Preserve color relationships from color_contrast selection"
            ]
        },
        "vocabulary_components": {
            "physics_terms": extract_physics_vocabulary(parameters["splash_details"]),
            "morphology_terms": extract_morphology_vocabulary(parameters["splash_details"]),
            "visual_characteristics": parameters["splash_details"]["visual_characteristics"],
            "compositional_descriptors": extract_compositional_vocabulary(parameters)
        }
    }
    
    return json.dumps(result, indent=2)


def extract_physics_vocabulary(splash_details: Dict) -> List[str]:
    """Helper: Extract physics-based vocabulary terms"""
    terms = []
    physics = splash_details["physics"]
    
    velocity = physics["optimal_velocity"]
    if velocity < 2:
        terms.append("low-energy impact")
    elif velocity < 7:
        terms.append("medium-velocity collision")
    else:
        terms.append("high-speed impact")
    
    if physics.get("surface_tension_critical"):
        terms.append("surface tension effects")
        terms.append("capillary wave formation")
    
    if physics.get("pressure_driven"):
        terms.append("pressurized stream")
        terms.append("directional jet")
    
    return terms


def extract_morphology_vocabulary(splash_details: Dict) -> List[str]:
    """Helper: Extract morphology-based vocabulary terms"""
    terms = []
    morph = splash_details["morphology"]
    
    symmetry = morph["symmetry"]
    terms.append(f"{symmetry.replace('_', ' ')} pattern")
    
    if "crown" in morph:
        terms.append("corona formation")
        terms.append("radial rim structure")
    
    if "strand" in str(morph):
        terms.append("ligament formation")
        terms.append("viscous strands")
    
    if "droplet" in str(morph):
        terms.append("droplet ejection")
        terms.append("ballistic trajectories")
    
    return terms


def extract_compositional_vocabulary(parameters: Dict) -> Dict[str, List[str]]:
    """Helper: Extract compositional vocabulary from all parameters"""
    comp = parameters["compositional_parameters"]
    
    return {
        "color": comp["color_contrast"]["details"]["visual_effect"].split(", "),
        "density": [comp["density_distribution"]["details"]["visual_effect"]],
        "scale": [comp["scale_hierarchy"]["details"]["visual_effect"]],
        "motion": [comp["motion_capture"]["details"]["visual_effect"]],
        "direction": [comp["directional_bias"]["details"]["visual_balance"]]
    }


# ============================================================================
# PHASE 2.6: Rhythmic Composition Tools (0 tokens)
# ============================================================================

@mcp.tool()
def get_splash_canonical_states() -> str:
    """
    List all 7 canonical splash states with their 5D parameter coordinates.

    Layer 1: Pure taxonomy lookup (0 tokens).

    Returns:
        JSON with state IDs, coordinates, and nearest visual type for each
        canonical splash configuration.
    """
    result = {}
    for sid, coords in SPLASH_CANONICAL_STATES.items():
        vtype, vdist = _splash_nearest_visual_type(coords)
        result[sid] = {
            "coordinates": coords,
            "nearest_visual_type": vtype,
            "visual_distance": round(vdist, 4),
        }
    return json.dumps({
        "canonical_states": result,
        "parameter_names": SPLASH_PARAMETER_NAMES,
        "dimensionality": SPLASH_DIMENSIONALITY,
        "total_states": len(result),
    }, indent=2)


@mcp.tool()
def get_splash_visual_types() -> str:
    """
    List all 5 splash visual types with keywords and optical properties.

    Layer 1: Pure taxonomy lookup (0 tokens).

    Visual types are nearest-neighbor targets used for vocabulary extraction.
    Each type has center coordinates, 5 image-generation keywords, and
    optical finish/scatter/transparency properties.

    Returns:
        JSON with all visual type specifications.
    """
    result = {}
    for vid, vdata in SPLASH_VISUAL_TYPES.items():
        result[vid] = {
            "center": vdata["center"],
            "keywords": vdata["keywords"],
            "optical": vdata["optical"],
        }
    return json.dumps({
        "visual_types": result,
        "total_types": len(result),
        "parameter_names": SPLASH_PARAMETER_NAMES,
    }, indent=2)


@mcp.tool()
def list_splash_rhythmic_presets() -> str:
    """
    List all 5 Phase 2.6 rhythmic presets for splash aesthetics.

    Layer 2: Pure lookup (0 tokens).

    Presets oscillate between two canonical splash states, creating
    temporal aesthetic composition suitable for animation keyframes,
    storyboard sequences, and multi-domain compositional limit cycles.

    Available presets:
        energy_surge (14):          drip ↔ explosion
        viscosity_sweep (22):       water ↔ paint
        freeze_thaw (18):           flow ↔ frozen crown
        scatter_convergence (26):   focused crown ↔ chaotic spray
        chromatic_burst (30):       muted drip ↔ vivid paint

    Returns:
        JSON with preset names, periods, patterns, and descriptions.
    """
    result = {}
    for name, cfg in SPLASH_RHYTHMIC_PRESETS.items():
        result[name] = {
            "period": cfg["period"],
            "state_a": cfg["state_a"],
            "state_b": cfg["state_b"],
            "pattern": cfg["pattern"],
            "description": cfg["description"],
        }
    return json.dumps({
        "presets": result,
        "total_presets": len(result),
        "periods": sorted(set(c["period"] for c in SPLASH_RHYTHMIC_PRESETS.values())),
    }, indent=2)


@mcp.tool()
def apply_splash_rhythmic_preset(preset_name: str) -> str:
    """
    Apply a curated splash rhythmic pattern preset.

    Layer 2: Deterministic sequence generation (0 tokens).

    Generates a complete oscillation sequence showing how the splash
    aesthetic evolves over one full period. Each step contains 5D
    parameter coordinates and nearest visual type classification.

    Args:
        preset_name: One of energy_surge, viscosity_sweep, freeze_thaw,
                     scatter_convergence, chromatic_burst

    Returns:
        JSON with complete oscillation sequence.
    """
    if preset_name not in SPLASH_RHYTHMIC_PRESETS:
        return json.dumps({
            "error": f"Unknown preset: {preset_name}",
            "available": list(SPLASH_RHYTHMIC_PRESETS.keys()),
        })

    cfg = SPLASH_RHYTHMIC_PRESETS[preset_name]
    state_a = SPLASH_CANONICAL_STATES[cfg["state_a"]]
    state_b = SPLASH_CANONICAL_STATES[cfg["state_b"]]
    period = cfg["period"]

    sequence = []
    for step in range(period):
        t = step / period
        state = _splash_interpolate_states(state_a, state_b, t, cfg["pattern"])
        vtype, vdist = _splash_nearest_visual_type(state)
        sequence.append({
            "step": step,
            "phase": round(t, 4),
            "state": {k: round(v, 4) for k, v in state.items()},
            "nearest_visual_type": vtype,
            "visual_distance": round(vdist, 4),
        })

    return json.dumps({
        "preset": preset_name,
        "period": period,
        "pattern": cfg["pattern"],
        "state_a": cfg["state_a"],
        "state_b": cfg["state_b"],
        "description": cfg["description"],
        "sequence": sequence,
        "total_steps": len(sequence),
    }, indent=2)


@mcp.tool()
def generate_splash_rhythmic_sequence(
    state_a_id: str,
    state_b_id: str,
    oscillation_pattern: str = "sinusoidal",
    num_cycles: int = 3,
    steps_per_cycle: int = 20,
    phase_offset: float = 0.0,
) -> str:
    """
    Generate custom rhythmic oscillation between any two splash states.

    Layer 2: Temporal composition (0 tokens).

    Create periodic transitions cycling between splash configurations.
    Useful for building new compositional presets or exploring the
    full range of fluid dynamics aesthetics.

    Args:
        state_a_id: Starting splash state (one of 7 canonical IDs)
        state_b_id: Alternating splash state
        oscillation_pattern: sinusoidal, triangular, or square
        num_cycles: Number of complete A→B→A cycles
        steps_per_cycle: Samples per cycle
        phase_offset: Starting phase (0.0 = A, 0.5 = B)

    Returns:
        JSON with sequence data, pattern info, and phase points.
    """
    if state_a_id not in SPLASH_CANONICAL_STATES:
        return json.dumps({
            "error": f"Unknown state: {state_a_id}",
            "available": list(SPLASH_CANONICAL_STATES.keys()),
        })
    if state_b_id not in SPLASH_CANONICAL_STATES:
        return json.dumps({
            "error": f"Unknown state: {state_b_id}",
            "available": list(SPLASH_CANONICAL_STATES.keys()),
        })

    state_a = SPLASH_CANONICAL_STATES[state_a_id]
    state_b = SPLASH_CANONICAL_STATES[state_b_id]
    total_steps = num_cycles * steps_per_cycle

    sequence = []
    for step in range(total_steps):
        raw_t = step / steps_per_cycle + phase_offset
        t = raw_t % 1.0
        state = _splash_interpolate_states(state_a, state_b, t, oscillation_pattern)
        vtype, vdist = _splash_nearest_visual_type(state)
        sequence.append({
            "step": step,
            "cycle": step // steps_per_cycle,
            "phase": round(t, 4),
            "state": {k: round(v, 4) for k, v in state.items()},
            "nearest_visual_type": vtype,
            "visual_distance": round(vdist, 4),
        })

    return json.dumps({
        "state_a": state_a_id,
        "state_b": state_b_id,
        "pattern": oscillation_pattern,
        "num_cycles": num_cycles,
        "steps_per_cycle": steps_per_cycle,
        "phase_offset": phase_offset,
        "total_steps": total_steps,
        "sequence": sequence,
    }, indent=2)


@mcp.tool()
def compute_splash_distance(splash_id_1: str, splash_id_2: str) -> str:
    """
    Compute distance between two splash states in 5D parameter space.

    Layer 2: Pure distance computation (0 tokens).

    Args:
        splash_id_1: First splash state (canonical ID)
        splash_id_2: Second splash state (canonical ID)

    Returns:
        Euclidean distance and per-parameter breakdown.
    """
    if splash_id_1 not in SPLASH_CANONICAL_STATES:
        return json.dumps({"error": f"Unknown: {splash_id_1}", "available": list(SPLASH_CANONICAL_STATES.keys())})
    if splash_id_2 not in SPLASH_CANONICAL_STATES:
        return json.dumps({"error": f"Unknown: {splash_id_2}", "available": list(SPLASH_CANONICAL_STATES.keys())})

    a = SPLASH_CANONICAL_STATES[splash_id_1]
    b = SPLASH_CANONICAL_STATES[splash_id_2]
    diffs = {p: round(b[p] - a[p], 4) for p in SPLASH_PARAMETER_NAMES}
    dist = _splash_euclidean_distance(a, b)

    return json.dumps({
        "splash_id_1": splash_id_1,
        "splash_id_2": splash_id_2,
        "euclidean_distance": round(dist, 4),
        "per_parameter_diff": diffs,
    }, indent=2)


@mcp.tool()
def compute_splash_trajectory(
    start_splash_id: str,
    end_splash_id: str,
    num_steps: int = 20,
) -> str:
    """
    Compute smooth trajectory between two splash states in morphospace.

    Layer 2: Deterministic trajectory integration (0 tokens).

    Enables visualization of smooth fluid aesthetic transitions—e.g.,
    crown splash gradually becoming thrown paint.

    Args:
        start_splash_id: Starting canonical splash state
        end_splash_id: Target canonical splash state
        num_steps: Number of interpolation steps (default: 20)

    Returns:
        Trajectory with intermediate states, distance profile, and
        transition characteristics.
    """
    if start_splash_id not in SPLASH_CANONICAL_STATES:
        return json.dumps({"error": f"Unknown: {start_splash_id}", "available": list(SPLASH_CANONICAL_STATES.keys())})
    if end_splash_id not in SPLASH_CANONICAL_STATES:
        return json.dumps({"error": f"Unknown: {end_splash_id}", "available": list(SPLASH_CANONICAL_STATES.keys())})

    a = SPLASH_CANONICAL_STATES[start_splash_id]
    b = SPLASH_CANONICAL_STATES[end_splash_id]
    total_dist = _splash_euclidean_distance(a, b)

    trajectory = []
    for i in range(num_steps + 1):
        t = i / num_steps
        state = {p: round(a[p] + t * (b[p] - a[p]), 4) for p in SPLASH_PARAMETER_NAMES}
        vtype, vdist = _splash_nearest_visual_type(state)
        nearest_canon, cdist = _splash_nearest_canonical(state)
        trajectory.append({
            "step": i,
            "t": round(t, 4),
            "state": state,
            "nearest_visual_type": vtype,
            "visual_distance": round(vdist, 4),
            "nearest_canonical": nearest_canon,
            "canonical_distance": round(cdist, 4),
        })

    return json.dumps({
        "start": start_splash_id,
        "end": end_splash_id,
        "total_distance": round(total_dist, 4),
        "num_steps": num_steps,
        "trajectory": trajectory,
    }, indent=2)


@mcp.tool()
def extract_splash_visual_vocabulary(
    state: Optional[Dict[str, float]] = None,
    splash_id: Optional[str] = None,
    strength: float = 1.0,
) -> str:
    """
    Extract visual vocabulary from splash parameter coordinates.

    Layer 2: Deterministic vocabulary mapping (0 tokens).

    Maps a 5D parameter state to the nearest visual type and returns
    image-generation-ready keywords with optical properties.

    Args:
        state: Parameter coordinates dict. Provide either state or splash_id.
        splash_id: Canonical splash state to use as state source.
        strength: Keyword weight multiplier [0.0, 1.0] (default: 1.0)

    Returns:
        Nearest visual type, keywords, optical properties, and vocabulary.
    """
    if state is None and splash_id is None:
        return json.dumps({"error": "Provide either state or splash_id"})
    if splash_id is not None:
        if splash_id not in SPLASH_CANONICAL_STATES:
            return json.dumps({"error": f"Unknown: {splash_id}", "available": list(SPLASH_CANONICAL_STATES.keys())})
        state = SPLASH_CANONICAL_STATES[splash_id]

    # Validate keys
    for p in SPLASH_PARAMETER_NAMES:
        if p not in state:
            return json.dumps({"error": f"Missing parameter: {p}", "required": SPLASH_PARAMETER_NAMES})

    vtype, vdist = _splash_nearest_visual_type(state)
    vdata = SPLASH_VISUAL_TYPES[vtype]
    vocab = _splash_select_vocabulary(state)
    nearest_canon, cdist = _splash_nearest_canonical(state)

    return json.dumps({
        "nearest_visual_type": vtype,
        "distance": round(vdist, 4),
        "keywords": vdata["keywords"],
        "optical_properties": vdata["optical"],
        "nearest_canonical_state": nearest_canon,
        "canonical_distance": round(cdist, 4),
        "vocabulary_by_category": vocab,
        "strength": strength,
        "input_state": {k: round(v, 4) for k, v in state.items()},
    }, indent=2)


# ============================================================================
# PHASE 2.7: Attractor Visualization Prompt Generation (0 tokens)
# ============================================================================

@mcp.tool()
def list_splash_attractor_presets() -> str:
    """
    List all 7 discovered/curated attractor presets for splash visualization.

    Layer 2: Pure lookup (0 tokens).

    Attractor presets represent multi-domain emergent states mapped to
    splash parameter coordinates. Each encodes the fluid-dynamics aesthetic
    that arises when the system locks into that attractor period during
    multi-domain composition.

    Returns:
        JSON catalog with names, basin sizes, classifications, and states.
    """
    result = {}
    for aid, adata in SPLASH_ATTRACTOR_PRESETS.items():
        vtype, vdist = _splash_nearest_visual_type(adata["state"])
        result[aid] = {
            "name": adata["name"],
            "description": adata["description"],
            "basin_size": adata["basin_size"],
            "classification": adata["classification"],
            "source_domains": adata["source_domains"],
            "state": adata["state"],
            "nearest_visual_type": vtype,
            "visual_distance": round(vdist, 4),
        }
    return json.dumps({
        "attractor_presets": result,
        "total_presets": len(result),
        "parameter_names": SPLASH_PARAMETER_NAMES,
    }, indent=2)


@mcp.tool()
def generate_splash_attractor_prompt(
    attractor_id: str = "",
    custom_state: Optional[Dict[str, float]] = None,
    mode: str = "composite",
    style_modifier: str = "",
    keyframe_count: int = 4,
) -> str:
    """
    Generate image generation prompt from attractor state or custom coordinates.

    Layer 2: Deterministic prompt synthesis (0 tokens).

    Translates splash aesthetic coordinates into visual prompts suitable
    for ComfyUI, Stable Diffusion, DALL-E, etc.

    Modes:
        composite:  Single blended prompt from attractor state
        split_view: Separate prompt per vocabulary category
        sequence:   Multiple keyframe prompts from nearest rhythmic preset

    Args:
        attractor_id: Preset attractor name (period_30, period_19, etc.)
            Use "" with custom_state for arbitrary coordinates.
        custom_state: Optional custom parameter coordinates dict.
            Overrides attractor_id if provided.
        mode: composite | split_view | sequence
        style_modifier: Optional prefix (e.g. "photorealistic", "fashion editorial")
        keyframe_count: Number of keyframes for sequence mode (default: 4)

    Returns:
        Dict with prompt(s), vocabulary details, and attractor metadata.
    """
    # ── Resolve state ─────────────────────────────────────────────────
    attractor_meta = None
    if custom_state is not None:
        for p in SPLASH_PARAMETER_NAMES:
            if p not in custom_state:
                return json.dumps({"error": f"Missing parameter: {p}", "required": SPLASH_PARAMETER_NAMES})
        state = custom_state
    elif attractor_id in SPLASH_ATTRACTOR_PRESETS:
        attractor_meta = SPLASH_ATTRACTOR_PRESETS[attractor_id]
        state = attractor_meta["state"]
    else:
        return json.dumps({
            "error": f"Provide attractor_id or custom_state",
            "available_attractors": list(SPLASH_ATTRACTOR_PRESETS.keys()),
        })

    # ── Vocabulary extraction ─────────────────────────────────────────
    vtype, vdist = _splash_nearest_visual_type(state)
    vdata = SPLASH_VISUAL_TYPES[vtype]
    vocab = _splash_select_vocabulary(state)
    nearest_canon, cdist = _splash_nearest_canonical(state)
    prefix = f"{style_modifier}, " if style_modifier else ""

    # ── COMPOSITE mode ────────────────────────────────────────────────
    if mode == "composite":
        all_keywords = list(vdata["keywords"])
        for cat_terms in vocab.values():
            all_keywords.extend(cat_terms)
        # Deduplicate preserving order
        seen = set()
        unique = []
        for kw in all_keywords:
            if kw not in seen:
                seen.add(kw)
                unique.append(kw)
        prompt = prefix + ", ".join(unique)

        return json.dumps({
            "mode": "composite",
            "prompt": prompt,
            "nearest_visual_type": vtype,
            "visual_distance": round(vdist, 4),
            "nearest_canonical_state": nearest_canon,
            "optical_properties": vdata["optical"],
            "attractor": {
                "id": attractor_id or "custom",
                "name": attractor_meta["name"] if attractor_meta else "Custom State",
                "basin_size": attractor_meta["basin_size"] if attractor_meta else None,
                "classification": attractor_meta["classification"] if attractor_meta else "custom",
            },
            "state": {k: round(v, 4) for k, v in state.items()},
        }, indent=2)

    # ── SPLIT_VIEW mode ───────────────────────────────────────────────
    elif mode == "split_view":
        panels = {}
        for category, terms in vocab.items():
            cat_prompt = prefix + ", ".join(terms)
            panels[category] = {"prompt": cat_prompt, "terms": terms}

        return json.dumps({
            "mode": "split_view",
            "panels": panels,
            "nearest_visual_type": vtype,
            "visual_distance": round(vdist, 4),
            "optical_properties": vdata["optical"],
            "attractor": {
                "id": attractor_id or "custom",
                "name": attractor_meta["name"] if attractor_meta else "Custom State",
                "classification": attractor_meta["classification"] if attractor_meta else "custom",
            },
            "state": {k: round(v, 4) for k, v in state.items()},
        }, indent=2)

    # ── SEQUENCE mode ─────────────────────────────────────────────────
    elif mode == "sequence":
        # Find nearest rhythmic preset by period proximity
        best_preset = None
        best_period_diff = float("inf")
        target_period = None

        if attractor_meta and attractor_meta.get("basin_size"):
            # Use attractor period as target
            period_str = attractor_id.replace("period_", "")
            try:
                target_period = int(period_str)
            except ValueError:
                target_period = None

        if target_period is None:
            target_period = 30  # Default to LCM hub

        for pname, pcfg in SPLASH_RHYTHMIC_PRESETS.items():
            diff = abs(pcfg["period"] - target_period)
            if diff < best_period_diff:
                best_period_diff = diff
                best_preset = pname

        # Generate keyframes from that preset
        cfg = SPLASH_RHYTHMIC_PRESETS[best_preset]
        state_a = SPLASH_CANONICAL_STATES[cfg["state_a"]]
        state_b = SPLASH_CANONICAL_STATES[cfg["state_b"]]
        period = cfg["period"]
        step_size = max(1, period // keyframe_count)

        keyframes = []
        for ki in range(keyframe_count):
            step = (ki * step_size) % period
            t = step / period
            kf_state = _splash_interpolate_states(state_a, state_b, t, cfg["pattern"])
            kf_vtype, kf_vdist = _splash_nearest_visual_type(kf_state)
            kf_vdata = SPLASH_VISUAL_TYPES[kf_vtype]
            kf_vocab = _splash_select_vocabulary(kf_state)

            all_kw = list(kf_vdata["keywords"])
            for cat_terms in kf_vocab.values():
                all_kw.extend(cat_terms)
            seen_kf = set()
            unique_kf = []
            for kw in all_kw:
                if kw not in seen_kf:
                    seen_kf.add(kw)
                    unique_kf.append(kw)

            keyframes.append({
                "keyframe": ki,
                "step": step,
                "phase": round(t, 4),
                "prompt": prefix + ", ".join(unique_kf),
                "nearest_visual_type": kf_vtype,
                "visual_distance": round(kf_vdist, 4),
                "state": {k: round(v, 4) for k, v in kf_state.items()},
            })

        return json.dumps({
            "mode": "sequence",
            "preset_used": best_preset,
            "preset_period": period,
            "keyframe_count": keyframe_count,
            "keyframes": keyframes,
            "attractor": {
                "id": attractor_id or "custom",
                "name": attractor_meta["name"] if attractor_meta else "Custom State",
                "classification": attractor_meta["classification"] if attractor_meta else "custom",
            },
        }, indent=2)

    else:
        return json.dumps({"error": f"Unknown mode: {mode}", "available": ["composite", "split_view", "sequence"]})


@mcp.tool()
def generate_splash_sequence_prompts(
    preset_name: str,
    keyframe_count: int = 4,
    style_modifier: str = "",
) -> str:
    """
    Generate keyframe prompts from a Phase 2.6 rhythmic preset.

    Layer 2: Deterministic keyframe extraction (0 tokens).

    Extracts evenly-spaced keyframes from a rhythmic oscillation
    and generates an image prompt for each. Useful for storyboards,
    animation keyframes, and multi-panel visualizations of temporal
    splash aesthetic evolution.

    Args:
        preset_name: One of the 5 rhythmic presets
        keyframe_count: Number of keyframes to extract (default: 4)
        style_modifier: Optional style prefix for all prompts

    Returns:
        Keyframes with step index, state, prompt, and vocabulary.
    """
    if preset_name not in SPLASH_RHYTHMIC_PRESETS:
        return json.dumps({"error": f"Unknown preset: {preset_name}", "available": list(SPLASH_RHYTHMIC_PRESETS.keys())})

    cfg = SPLASH_RHYTHMIC_PRESETS[preset_name]
    state_a = SPLASH_CANONICAL_STATES[cfg["state_a"]]
    state_b = SPLASH_CANONICAL_STATES[cfg["state_b"]]
    period = cfg["period"]
    step_size = max(1, period // keyframe_count)
    prefix = f"{style_modifier}, " if style_modifier else ""

    keyframes = []
    for ki in range(keyframe_count):
        step = (ki * step_size) % period
        t = step / period
        kf_state = _splash_interpolate_states(state_a, state_b, t, cfg["pattern"])
        kf_vtype, kf_vdist = _splash_nearest_visual_type(kf_state)
        kf_vdata = SPLASH_VISUAL_TYPES[kf_vtype]
        kf_vocab = _splash_select_vocabulary(kf_state)

        all_kw = list(kf_vdata["keywords"])
        for cat_terms in kf_vocab.values():
            all_kw.extend(cat_terms)
        seen = set()
        unique = []
        for kw in all_kw:
            if kw not in seen:
                seen.add(kw)
                unique.append(kw)

        keyframes.append({
            "keyframe": ki,
            "step": step,
            "phase": round(t, 4),
            "prompt": prefix + ", ".join(unique),
            "nearest_visual_type": kf_vtype,
            "visual_distance": round(kf_vdist, 4),
            "state": {k: round(v, 4) for k, v in kf_state.items()},
        })

    return json.dumps({
        "preset": preset_name,
        "period": period,
        "keyframe_count": keyframe_count,
        "keyframes": keyframes,
    }, indent=2)


@mcp.tool()
def map_splash_parameters(
    splash_id: str,
    intensity: str = "moderate",
    emphasis: str = "dynamics",
) -> str:
    """
    Map splash state to visual parameters for image generation.

    Layer 2: Deterministic operation (0 tokens).

    Args:
        splash_id: Canonical splash state ID
        intensity: subtle, moderate, or dramatic
        emphasis: dynamics, fragmentation, light, temporal, or color

    Returns:
        Complete parameter set for visual synthesis including vocabulary
        weighted by intensity and emphasis.
    """
    if splash_id not in SPLASH_CANONICAL_STATES:
        return json.dumps({"error": f"Unknown: {splash_id}", "available": list(SPLASH_CANONICAL_STATES.keys())})

    state = SPLASH_CANONICAL_STATES[splash_id]
    vtype, vdist = _splash_nearest_visual_type(state)
    vdata = SPLASH_VISUAL_TYPES[vtype]
    vocab = _splash_select_vocabulary(state)

    intensity_weights = {"subtle": 0.6, "moderate": 1.0, "dramatic": 1.5}
    weight = intensity_weights.get(intensity, 1.0)

    # Select emphasis category
    emphasis_map = {
        "dynamics": "fluid_dynamics",
        "fragmentation": "fragmentation",
        "light": "light_capture",
        "temporal": "temporal_quality",
        "color": "color_behavior",
    }
    primary_cat = emphasis_map.get(emphasis, "fluid_dynamics")
    primary_terms = vocab.get(primary_cat, [])

    return json.dumps({
        "splash_id": splash_id,
        "intensity": intensity,
        "emphasis": emphasis,
        "weight": weight,
        "state": state,
        "nearest_visual_type": vtype,
        "visual_distance": round(vdist, 4),
        "optical_properties": vdata["optical"],
        "primary_vocabulary": primary_terms,
        "full_vocabulary": vocab,
        "keywords": vdata["keywords"],
    }, indent=2)


# ============================================================================
# DOMAIN REGISTRY HELPER
# ============================================================================

# ============================================================================
# Phase 2.8: Aesthetic Decomposition (text → coordinates)
# ============================================================================

@mcp.tool()
def decompose_splash_from_description(description: str) -> str:
    """Decompose a text description into splash aesthetic coordinates.

    LAYER 2: Deterministic keyword matching (0 LLM tokens).

    Inverse of prompt generation: takes a text description (from Claude
    vision output, user description, or any text describing a splash
    aesthetic) and recovers the 5D splash coordinates by matching against
    the visual vocabulary types.

    Algorithm:
      1. Tokenize description into words and full lowercase text
      2. Score each visual type by keyword fragment matching + optical matching
      3. Softmax scores → blending weights
      4. Weighted average of visual type centers → 5D coordinates
      5. Confidence = max_score / max_possible_score

    This completes the round-trip:
      coordinates → prompt → image → description → coordinates

    Args:
        description: Text describing a splash aesthetic. Can be a Claude
            vision description of an image, a user's natural language
            description, or generated prompt text. Longer, more specific
            descriptions yield higher confidence and accuracy.

    Returns:
        JSON with:
        - coordinates: 5D splash parameter values {impact_energy, ...}
        - confidence: 0-1 detection strength (how much splash vocabulary found)
        - nearest_type: Best-matching visual type ID
        - nearest_type_distance: Euclidean distance from blend to nearest center
        - type_scores: Raw match score per visual type
        - type_weights: Softmax blending weights used
        - matched_fragments: Which keyword fragments were found in text
        - optical_matches: Which optical properties matched
        - domain_detected: Whether confidence exceeds minimum threshold

    Cost: 0 tokens (pure Layer 2 deterministic computation)

    Example:
        >>> decompose_splash_from_description(
        ...     "frozen moment of milk droplet impact forming perfect crown, "
        ...     "razor-sharp corona rim with satellite droplets"
        ... )
        {
            "coordinates": {"impact_energy": 0.55, "temporal_freeze": 0.95, ...},
            "confidence": 0.42,
            "nearest_type": "frozen_crown",
            "domain_detected": true
        }
    """
    # Tokenize
    lower = description.lower()
    words = set(re.findall(r'[a-z]+(?:-[a-z]+)*', lower))

    # Score each visual type
    type_scores = {}
    all_matched = []
    optical_matches = {}

    for vid, vdata in SPLASH_VISUAL_TYPES.items():
        score, matched = _splash_score_visual_type(vdata, words, lower)
        type_scores[vid] = score
        all_matched.extend(matched)

        # Collect optical matches
        for m in matched:
            if m.startswith("optical:"):
                val = m.split(":", 1)[1]
                for prop_name, prop_value in vdata.get("optical", {}).items():
                    if prop_value == val:
                        optical_matches[prop_name] = val

    # Confidence
    max_score = max(type_scores.values()) if type_scores else 0
    # Max possible: 5 keywords × 1.0 + 3 optical × 0.5 = 6.5
    max_possible = 5 * 1.0 + 3 * 0.5
    confidence = min(1.0, max_score / max_possible) if max_possible > 0 else 0.0
    min_threshold = 0.05

    if confidence < min_threshold:
        return json.dumps({
            "coordinates": {p: 0.5 for p in SPLASH_PARAMETER_NAMES},
            "confidence": 0.0,
            "nearest_type": "",
            "nearest_type_distance": None,
            "type_scores": {k: round(v, 3) for k, v in type_scores.items()},
            "type_weights": {},
            "matched_fragments": [],
            "optical_matches": {},
            "domain_detected": False,
        }, indent=2)

    # Blend coordinates
    weights = _splash_softmax(type_scores)
    coordinates = _splash_blend_coordinates(weights)

    # Find nearest visual type
    nearest_type = max(type_scores, key=type_scores.get)
    nearest_center = SPLASH_VISUAL_TYPES[nearest_type]["center"]
    nearest_distance = _splash_euclidean_distance(coordinates, nearest_center)

    # Deduplicate matched fragments (exclude optical/color prefixed ones for display)
    unique_matched = list(dict.fromkeys(
        m for m in all_matched if not m.startswith(("optical:", "color:"))
    ))

    return json.dumps({
        "coordinates": {k: round(v, 4) for k, v in coordinates.items()},
        "confidence": round(confidence, 4),
        "nearest_type": nearest_type,
        "nearest_type_distance": round(nearest_distance, 4),
        "type_scores": {k: round(v, 3) for k, v in type_scores.items()},
        "type_weights": {k: round(v, 4) for k, v in weights.items()},
        "matched_fragments": unique_matched,
        "optical_matches": optical_matches,
        "domain_detected": True,
    }, indent=2)


@mcp.tool()
def get_splash_domain_registry_config() -> str:
    """
    Get complete domain configuration for emergent attractor discovery.

    Returns the data needed by domain_registry.py to integrate splash
    aesthetics into the Tier 4D compositional limit cycle discovery system.
    Follows the ADDING_NEW_DOMAINS.md integration pattern.

    Returns:
        JSON with domain_id, parameter_names, state_coordinates, presets,
        vocabulary, periods, and attractor_presets.
    """
    # Build presets in registry format
    registry_presets = {}
    for name, cfg in SPLASH_RHYTHMIC_PRESETS.items():
        registry_presets[name] = {
            "name": name,
            "period": cfg["period"],
            "state_a_id": cfg["state_a"],
            "state_b_id": cfg["state_b"],
            "pattern": cfg["pattern"],
            "description": cfg["description"],
        }

    # Build vocabulary from visual types
    registry_vocab = {}
    for vtype, vdata in SPLASH_VISUAL_TYPES.items():
        registry_vocab[vtype] = vdata["keywords"]

    return json.dumps({
        "domain_id": "splash",
        "display_name": "Splash Aesthetics",
        "description": "Physics-based fluid dynamics splash composition",
        "mcp_server": "splash-aesthetics-mcp",
        "parameter_names": SPLASH_PARAMETER_NAMES,
        "state_coordinates": SPLASH_CANONICAL_STATES,
        "presets": registry_presets,
        "vocabulary": registry_vocab,
        "periods": sorted(set(c["period"] for c in SPLASH_RHYTHMIC_PRESETS.values())),
        "attractor_presets": {
            aid: {
                "name": adata["name"],
                "basin_size": adata["basin_size"],
                "classification": adata["classification"],
                "state": adata["state"],
            }
            for aid, adata in SPLASH_ATTRACTOR_PRESETS.items()
        },
        "domain_registry_ready": True,
    }, indent=2)


# ============================================================================
# SERVER INFO (updated with Phase 2.6/2.7 capabilities)
# ============================================================================

@mcp.tool()
def get_server_info() -> str:
    """
    Get information about the Splash Aesthetics MCP server.

    Returns server metadata, architecture overview, Phase 2.6/2.7
    capabilities, and usage guidance.
    """
    return json.dumps({
        "name": "Splash Aesthetics MCP",
        "version": "2.0.0-phase2.7+tier4d",
        "architecture": "three_layer_olog",
        "description": (
            "Physics-based splash composition with zero-cost enhancement. "
            "Phase 2.6 rhythmic presets for temporal composition, "
            "Phase 2.7 attractor visualization for multi-domain emergence."
        ),
        "layers": {
            "layer_1": "Pure taxonomy (splash types, phases, parameters, visual types)",
            "layer_2": (
                "Deterministic mapping (intent classification, parameter selection, "
                "rhythmic presets, trajectory computation, distance, vocabulary, "
                "attractor prompts)"
            ),
            "layer_3": "Claude synthesis interface (structured data for creative composition)",
        },
        "domain_coverage": {
            "splash_types": 7,
            "temporal_phases": 6,
            "compositional_parameters": 6,
            "canonical_states": len(SPLASH_CANONICAL_STATES),
            "visual_types": len(SPLASH_VISUAL_TYPES),
            "parameter_dimensions": SPLASH_DIMENSIONALITY,
        },
        "phase_2_6_enhancements": {
            "rhythmic_presets": True,
            "preset_count": len(SPLASH_RHYTHMIC_PRESETS),
            "periods": sorted(set(c["period"] for c in SPLASH_RHYTHMIC_PRESETS.values())),
            "preset_names": list(SPLASH_RHYTHMIC_PRESETS.keys()),
            "custom_oscillation": True,
            "trajectory_computation": True,
            "patterns": ["sinusoidal", "triangular", "square"],
        },
        "phase_2_7_enhancements": {
            "attractor_visualization": True,
            "attractor_presets": list(SPLASH_ATTRACTOR_PRESETS.keys()),
            "prompt_modes": ["composite", "split_view", "sequence"],
            "visual_vocabulary": True,
            "domain_registry_ready": True,
        },
        "physics_foundation": [
            "Fluid dynamics",
            "Surface tension effects",
            "Impact mechanics",
            "Droplet breakup",
            "Ballistic trajectories",
        ],
        "aesthetic_applications": [
            "Fashion editorial (Nick Knight style)",
            "Product photography",
            "High-speed capture",
            "Abstract expressionism",
            "Scientific visualization",
        ],
        "functorial_connections": [
            "Temporal phases → Catastrophe theory",
            "Color contrast → Heraldic tinctures",
            "Scale hierarchy → Photographic perspective",
        ],
        "compatible_servers": [
            "catastrophe-morph-mcp",
            "diatom-morphology-mcp",
            "surface-design-aesthetics",
            "microscopy-aesthetics-mcp",
            "aesthetic-dynamics-core",
            "composition-graph-mcp",
        ],
        "cost_model": "0 tokens for Layers 1-2, creative synthesis by Claude at Layer 3",
    }, indent=2)


if __name__ == "__main__":
    # Run server locally
    mcp.run()
