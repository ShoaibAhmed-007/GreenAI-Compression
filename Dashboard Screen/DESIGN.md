# Design System Specification: Nature-Tech Editorial

## 1. Overview & Creative North Star: "The Living Laboratory"
This design system rejects the sterile, "plastic" aesthetic of traditional SaaS. Our Creative North Star is **The Living Laboratory**—a space where high-precision machine learning meets the fluid, organic pulse of the natural world. 

We move beyond the "template" look by embracing **Intentional Asymmetry** and **Tonal Depth**. Instead of rigid, boxed-in layouts, we use overlapping elements, sophisticated gradients, and high-contrast typography scales to create a digital experience that feels grown, not just built. We celebrate the "technical" through JetBrains Mono but ground it in the "organic" through Space Grotesk and a lush, verdant palette.

---

## 2. Colors & Surface Philosophy
The palette is rooted in the deep forest and rising mist. We use light and shadow—rather than lines—to define our world.

### The "No-Line" Rule
**Explicit Instruction:** Prohibit 1px solid borders for sectioning. Boundaries must be defined solely through background color shifts or subtle tonal transitions. A `surface-container-low` section sitting on a `surface` background provides all the separation the eye needs.

### Surface Hierarchy & Nesting
Treat the UI as physical layers—stacked sheets of frosted glass.
- **Base Layer:** `surface` (#121411)
- **Primary Containers:** `surface-container` (#1e201d)
- **Emphasis Containers:** `surface-container-high` (#292b27)
- **Nested Content:** Use `surface-container-low` (#1a1c19) inside higher tiers to create "recessed" wells for data.

### The "Glass & Gradient" Rule
To achieve a "signature" feel, floating elements (modals, dropdowns, floating nav) must use **Glassmorphism**:
- **Fill:** `surface_variant` at 60% opacity.
- **Effect:** `backdrop-filter: blur(12px)`.
- **Gradients:** Use a linear gradient from `primary` (#5bdda8) to `primary_container` (#005c40) at a 135° angle for high-impact CTAs and hero backgrounds to provide visual "soul."

---

## 3. Typography: The Editorial Voice
Our typography balances the raw energy of tech with the clarity of sustainability reporting.

| Level | Token | Font Family | Size | Character |
| :--- | :--- | :--- | :--- | :--- |
| **Display** | `display-lg` | Space Grotesk | 3.5rem | Bold, urgent, architectural. |
| **Headline** | `headline-md` | Space Grotesk | 1.75rem | Assertive and modern. |
| **Title** | `title-md` | Plus Jakarta Sans | 1.125rem | Friendly, professional, legible. |
| **Body** | `body-md` | Plus Jakarta Sans | 0.875rem | High readability for long-form data. |
| **Technical** | `label-md` | JetBrains Mono | 0.75rem | Precise, monospaced for metrics. |

**Hierarchy Note:** Use Space Grotesk for all "storytelling" moments and Plus Jakarta Sans for "actionable" moments. Technical data (latencies, weights, CO2 offsets) must always use JetBrains Mono to signal mathematical precision.

---

## 4. Elevation & Depth: Tonal Layering
We do not use shadows to simulate height; we use color to simulate presence.

- **The Layering Principle:** Depth is achieved by stacking. Place a `surface-container-lowest` card on a `surface-container-low` section to create a soft, natural lift.
- **Ambient Shadows:** For floating components, use an "Atmospheric Shadow." 
    - `box-shadow: 0 20px 40px rgba(0, 0, 0, 0.25);`
    - Shadow color should never be pure black; it should be a deep, tinted version of `surface_container_lowest`.
- **The Ghost Border:** If a boundary is strictly required for accessibility, use a "Ghost Border": `outline-variant` (#42493e) at **15% opacity**. Never use 100% opaque borders.

---

## 5. Components

### Buttons: The Action Catalyst
- **Primary:** Gradient fill (`primary` to `primary_container`), `on_primary` text. No border. `xl` roundedness (0.75rem).
- **Secondary:** `secondary_container` fill. Subtle hover lift (2px Y-axis).
- **Technical/Tertiary:** `outline_variant` (15% opacity) with JetBrains Mono text for "DevOps" style actions.

### Cards & Data Surfaces
- **Rules:** No dividers. Use `32px` vertical padding (from spacing scale) to separate content.
- **Data Tables:** Right-align all JetBrains Mono technical data to ensure decimal points align visually. Use `surface-container-low` for alternating "zebra" backgrounds instead of lines.

### Expressive Charts (Recharts Style)
- **Stroke:** Use `primary` (#5bdda8) for active trends.
- **Area Fills:** Use a gradient of `primary` to transparent (0% opacity).
- **Grid Lines:** Use `outline_variant` at 10% opacity; horizontal lines only.

### Status Indicators: Model Readiness
- **Active:** Pulsing `primary` (#5bdda8) dot with a soft outer glow.
- **Training:** `secondary` (#afd09a) with a subtle "breathing" opacity animation (100% to 60%).
- **Error:** `error` (#ffb4ab) container with `on_error_container` text.

---

## 6. Do’s and Don’ts

### Do
- **Do** use organic, asymmetric container shapes (e.g., one corner with `xl` radius, others with `md`) for hero sections.
- **Do** use large amounts of negative space to convey "clean energy."
- **Do** use `JetBrains Mono` for any number that can be measured or calculated.

### Don't
- **Don't** use purple, pink, or saturated "tech blues." Stay within the forest/teal/mint spectrum.
- **Don't** use standard `1px` borders to separate cards. Use background tonal shifts.
- **Don't** use sharp `0px` corners unless it is for a specific "Brutalist" technical terminal component.
- **Don't** use drop shadows on flat UI surfaces; only use them for elements that truly "float" (modals, tooltips).