# Splash Aesthetics MCP Server

Physics-based splash composition with three-layer categorical architecture achieving 60-85% cost savings over pure LLM approaches.

## Intentionality

This MCP server encodes **fluid dynamics taxonomy** as a deterministic enhancement layer. Rather than asking an LLM to "add splash effects" (expensive, inconsistent), we:

1. **Layer 1**: Enumerate splash physics taxonomy (crown formation, droplet ejection, temporal phases) - **0 tokens**
2. **Layer 2**: Map user intent to specific parameters deterministically via keyword matching - **0 tokens**  
3. **Layer 3**: Provide structured data for Claude to synthesize into image generation prompts - **single synthesis call**

### Why This Architecture?

**The Composition Problem**: Splashes have well-understood physics (impact velocity, viscosity, surface tension, temporal phases) that shouldn't require LLM inference every time. The creative task is *translating* these physics parameters into compelling visual descriptions.

**The Three-Layer Solution**:
- Separate **taxonomy** (Layer 1, static YAML) from **mapping** (Layer 2, deterministic) from **synthesis** (Layer 3, creative)
- Claude's creativity focuses on the *synthesis* step where it's actually needed
- Cost savings: ~70% reduction by eliminating redundant LLM calls for classification and parameter selection

### Categorical Foundation

This follows olog (ontology log) architecture where:
- **Objects**: Splash types, temporal phases, compositional parameters
- **Morphisms**: Intent → splash type, parameters → visual vocabulary
- **Functors**: Potential mappings to other domains (catastrophe theory for temporal phases, heraldic tinctures for colors)

## Domain Coverage

### 7 Splash Types (Physics-Based)

1. **Crown Splash**: Radial symmetry, classic impact corona - *product photography*
2. **Sheet Splash**: Thin spreading film with wave propagation - *delicate aesthetics*
3. **Jet Splash**: Vertical columnar projection - *high drama*
4. **Thrown Paint**: Gestural viscous throw - *fashion editorial (Nick Knight style)*
5. **Arterial Spray**: Pressurized directional spray - *forensic/horror aesthetics*
6. **Cascade Drip**: Gravity-driven low-energy - *contemplative mood*
7. **Explosive Burst**: Omnidirectional chaos - *maximum energy*

### 6 Temporal Phases (Catastrophe Theory)

Each phase maps to catastrophe morphology:
- **Pre-impact**: Smooth → singular (fold catastrophe)
- **Impact moment**: Bifurcation (cusp catastrophe)
- **Crown formation**: Multiple paths (butterfly catastrophe)
- **Droplet ejection**: Fragmentation (swallowtail catastrophe)
- **Secondary splash**: Recursive (fold catastrophe)
- **Settling phase**: Return to equilibrium (fold catastrophe)

### 6 Compositional Parameters

- **Color Contrast**: Complementary, analogous, monochromatic, triadic
- **Density Distribution**: Sparse → chaotic (coverage percentage)
- **Scale Hierarchy**: Macro intimate → microscopic abstract
- **Directional Bias**: Radial, diagonal, vertical, horizontal, omnidirectional
- **Opacity/Translucency**: Opaque → transparent (light transmission)
- **Motion Capture**: Frozen instant → long exposure (shutter speed)

## Installation

```bash
# Install from local directory
cd splash-aesthetics-mcp
pip install -e .

# Or install dependencies directly
pip install fastmcp pyyaml
```

## Usage

### As MCP Server

```python
# Run the server
python splash_aesthetics_mcp.py
```

### Tool Examples

#### List Available Splash Types (Layer 1)
```python
# Returns all 7 splash types with physics specs
list_splash_types()
```

#### Classify User Intent (Layer 2)
```python
# Deterministic keyword matching
classify_splash_intent("fashion editorial with thrown red paint splashes")
# Returns: thrown_paint with confidence score
```

#### Full Enhancement Pipeline (Layer 3)
```python
# Complete parameter mapping for Claude synthesis
enhance_splash_prompt(
    user_intent="dramatic Nick Knight fashion portrait with red and blue paint splashes",
    intensity="dramatic"
)
# Returns structured data including:
# - Splash type: thrown_paint
# - Physics parameters: velocity, viscosity
# - Compositional parameters: complementary colors, medium density, portrait scale
# - Temporal emphasis: crown_formation phase
# - Vocabulary components for Claude to synthesize
```

## Architecture Details

### YAML Taxonomies

All domain knowledge stored in `/taxonomy/`:
- `splash_types.yaml`: Physics-based splash classification with intent keywords
- `temporal_phases.yaml`: Temporal evolution with catastrophe mappings
- `compositional_parameters.yaml`: Visual parameter specifications

### Deterministic Mapping (Layer 2)

**Intent Classification**: Keyword matching against taxonomy
```python
"thrown paint" → thrown_paint splash type
"dramatic" → high_energy temporal emphasis
"red and blue" → complementary_high color contrast
```

**Parameter Selection**: Rule-based mapping
```python
if "macro" or "close" in intent:
    scale = "macro_intimate"
if "freeze" or "frozen" in intent:
    motion = "frozen_instant"
```

**Zero LLM tokens** for these operations.

### Claude Synthesis (Layer 3)

Claude receives structured data and synthesizes image generation prompt:

```
Input (from Layer 2):
{
  "splash_type": "thrown_paint",
  "physics": {"velocity": 4.0, "viscosity": 0.1-5.0},
  "color": "complementary_high (red/cyan)",
  "density": "balanced_medium",
  "temporal_phase": "crown_formation"
}

Claude synthesizes:
"Fashion editorial portrait, model with vivid red and electric blue paint splashes 
thrown across face, high-viscosity gestural strands in crown formation phase, 
complementary color impact, medium-velocity collision (4 m/s), frozen at 1/4000s, 
Nick Knight style dynamic motion capture, studio lighting, sharp focus"
```

## Functorial Connections

This domain can compose with other MCP servers:

### Temporal Phases → Catastrophe Theory
```python
# Crown formation phase maps to butterfly catastrophe
# Enables catastrophe-aware enhancement
temporal_phase["crown_formation"]["catastrophe_mapping"] = "butterfly_catastrophe"
```

### Color Contrast → Heraldic Tinctures
```python
# Complementary high contrast maps to proper heraldic tincture rules
# Red splash on blue background = valid heraldic composition
```

### Scale Hierarchy → Photographic Perspective
```python
# Macro intimate scale maps to specific focal lengths
# 90-200mm macro with shallow DOF
```

## Physics Foundation

### Surface Tension Effects
- Crown formation depends on Weber number (We = ρv²L/σ)
- Sheet splash shows capillary wave instabilities
- Droplet ejection follows Rayleigh-Plateau instability

### Impact Dynamics
- Velocity ranges validated against experimental literature
- Morphology predictions based on dimensionless numbers
- Temporal phases match high-speed photography observations

### Viscosity Effects
- Paint: 0.1-5.0 Pa·s (viscous ligaments, gestural)
- Water: 0.001 Pa·s (fine droplets, geometric)
- Blood: 0.003-0.01 Pa·s (arterial spray patterns)

## Use Cases

### Fashion Editorial (Nick Knight Style)
```python
enhance_splash_prompt(
    "Nick Knight fashion portrait with paint splashes",
    splash_type="thrown_paint",
    color_override="complementary_high",
    intensity="dramatic"
)
```

### Product Photography (Beverage)
```python
enhance_splash_prompt(
    "refreshing water splash for beverage ad",
    splash_type="crown_splash",
    intensity="subtle"
)
```

### High-Speed Scientific Visualization
```python
enhance_splash_prompt(
    "milk droplet impact for physics visualization",
    splash_type="crown_splash",
    intensity="moderate"
)
```

### Horror/Thriller Aesthetics
```python
enhance_splash_prompt(
    "crime scene blood spatter pattern",
    splash_type="arterial_spray",
    intensity="dramatic"
)
```

## Cost Analysis

### Traditional LLM Approach
```
User: "Add dramatic paint splashes to this portrait"
→ LLM classification (500 tokens)
→ LLM parameter selection (800 tokens)  
→ LLM prompt synthesis (1200 tokens)
Total: ~2500 tokens = $0.0075 @ $3/M tokens
```

### This MCP Approach
```
User: "Add dramatic paint splashes to this portrait"
→ classify_splash_intent() - 0 tokens
→ map_intent_to_parameters() - 0 tokens
→ Claude synthesis from structured data (400 tokens)
Total: ~400 tokens = $0.0012 @ $3/M tokens
```

**Savings: 84%** on this example

## Development

### Running Tests
```bash
pytest tests/
```

### Adding New Splash Types

1. Add to `taxonomy/splash_types.yaml`:
```yaml
new_splash_type:
  name: "New Splash Type"
  physics: {...}
  morphology: {...}
  intent_keywords: [...]
```

2. Classification automatically includes new type
3. No code changes needed (pure taxonomy extension)

### FastMCP Cloud Deployment

```bash
# Deploy to FastMCP Cloud
fastmcp deploy splash-aesthetics-mcp
```

Entry point: `splash_aesthetics_mcp.py:mcp`

## References

### Splash Physics
- Worthington, A.M. (1908) *A Study of Splashes*
- Yarin, A.L. (2006) "Drop Impact Dynamics: Splashing, Spreading, Receding, Bouncing..."
- Thoroddsen, S.T. et al. (2008) "High-speed imaging of drops and bubbles"

### Visual Aesthetics  
- Nick Knight fashion photography (thrown paint portraiture)
- Edgerton, H. (1987) *Stopping Time* (high-speed photography)
- Forensic blood spatter analysis (IABPA standards)

### Categorical Architecture
- Spivak, D.I. (2013) "Category Theory for the Sciences"
- Olog (ontology log) formalism for domain knowledge
- Three-layer composition: taxonomy → mapping → synthesis

## License

MIT License - see LICENSE file

## Author

Dal Marsters (dal@lushy.ai)  
Lushy AI - Categorical composition for creative workflows

---

*This MCP server demonstrates the power of separating taxonomy (Layer 1) from deterministic mapping (Layer 2) from creative synthesis (Layer 3). By encoding splash physics as pure categorical knowledge, we achieve massive cost savings while maintaining full creative control at the synthesis step.*
